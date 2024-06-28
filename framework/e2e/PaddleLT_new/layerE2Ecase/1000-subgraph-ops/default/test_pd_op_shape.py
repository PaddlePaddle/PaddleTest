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



class PrimitiveOp_9b986e0dc26cdda02ec1a5915932720d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 24, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2659f44903f557ad75f462bf5347dcde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b986e0dc26cdda02ec1a5915932720d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


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


class PrimitiveOp_b3c51b83cc67661be238cfb3cf825e4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e04bdbda9f05d143439ca78c35b734f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3c51b83cc67661be238cfb3cf825e4e
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


class PrimitiveOp_376d3c9930ad9c649ff264780f663ad0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73e9d18e416b83443b6a59df2842d866(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376d3c9930ad9c649ff264780f663ad0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_44feff5769bb2f8dc541c859d5d3e142(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9566a5fcdbf35f0fbb0e75dc5b94825e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44feff5769bb2f8dc541c859d5d3e142
    def get_inputs(self):
        return [
            paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d8b6088acdf6636e5ec7167a352dab46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33aa3e7ff071a88934354e8819ca1463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b6088acdf6636e5ec7167a352dab46
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2cfa556999fc09aec69573a2f0f52ead(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 7, 7, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad7e3507d7a800d489c62b1cc83e2b8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfa556999fc09aec69573a2f0f52ead
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


class PrimitiveOp_8a63f7abf61748efdc20c140023e3118(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5b4b34e0894d172dd7fcd6989d9690a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a63f7abf61748efdc20c140023e3118
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


class TestPrimitiveOp_8c4fa9b5fbc95c39f9eb8931af734e5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44feff5769bb2f8dc541c859d5d3e142
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


class PrimitiveOp_a52bee9a71f461be1154464ebc88d522(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9af68f550dcf51f16615a79576ece200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a52bee9a71f461be1154464ebc88d522
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eeeeb16a22d04eb0ca7ce25bed965fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b6088acdf6636e5ec7167a352dab46
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13f10ad294b157d3cc202b77768c476f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376d3c9930ad9c649ff264780f663ad0
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_008c7df90bfafc172f40d98c1346fb09(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 14, 14, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa356ad48d7f408cbb260a61cdf77801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_008c7df90bfafc172f40d98c1346fb09
    def get_inputs(self):
        return [
            paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c28418a8577a881b8538277157a41f99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7bc5b591f4ba71d3865ee97d6b511e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c28418a8577a881b8538277157a41f99
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


class PrimitiveOp_4acda443423e75d640e40df381552696(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db99b1a55eb891492603967e7f7083ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4acda443423e75d640e40df381552696
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0970f9570a05caf47ec1e0978456dbd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 28, 28, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7f8fc7cc3e2b3447ddc64e91831e42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0970f9570a05caf47ec1e0978456dbd5
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7640572b2d01c641c85e4d0dd2b4f3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3c51b83cc67661be238cfb3cf825e4e
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


class PrimitiveOp_521b3f7dcef4851e16dc15cdcce6d398(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 20, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45a42098bfa895d6419b1ce1feea3fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_521b3f7dcef4851e16dc15cdcce6d398
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


class PrimitiveOp_dfbfcb481c8125039faa0c6b62aa8ab4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5f82fc93f9f6a74507888625d749e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfbfcb481c8125039faa0c6b62aa8ab4
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4271b3c81c66be9b2d472eb848d3f542(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 56, 56, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c8bdf3f237ded64a1f2c71ce58420fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4271b3c81c66be9b2d472eb848d3f542
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


class TestPrimitiveOp_dd93a0c8205de8527ca7d76481a754d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a52bee9a71f461be1154464ebc88d522
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


class PrimitiveOp_027515414d69a2c1a7da7da974b9e8ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_426a1461e90660ad1bf8438023fcbf9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_027515414d69a2c1a7da7da974b9e8ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0467a1e3b7727fd6d955583ed7600172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b6088acdf6636e5ec7167a352dab46
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


class PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e52b9ae5d6751c3f21daa9e63de633b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc
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


class PrimitiveOp_aa48f1814d3fc6596a08065500b7fb58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e307fcfd0b2ea3db73914086d674bbdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa48f1814d3fc6596a08065500b7fb58
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4e6cb7628b0a7cd95df7517b1c991f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b6088acdf6636e5ec7167a352dab46
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_023c7bfe4862c893333d1aa29c90792b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfa556999fc09aec69573a2f0f52ead
    def get_inputs(self):
        return [
            paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_607dbf3164b7f41fb31cb30558fb6210(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f705dea18f3ba4371bc60394a8371ba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_607dbf3164b7f41fb31cb30558fb6210
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_eed9d2e8f21b9208ba2629ea01774e22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2bd65f733c127ff4ddc406f2d56042b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eed9d2e8f21b9208ba2629ea01774e22
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


class PrimitiveOp_0e3fed05d5dbcacb0351ff9705267a1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c738c35aa4cab80163b5371acefa13e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e3fed05d5dbcacb0351ff9705267a1d
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


class PrimitiveOp_3923ecae0c120ba8da8690300f1a01fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_331a633bfcbab1574a7d4606e5ad925c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3923ecae0c120ba8da8690300f1a01fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95ff284d279583a1c6761f8e36f8966d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3923ecae0c120ba8da8690300f1a01fa
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


class TestPrimitiveOp_d5f82fc93f9f6a74507888625d749e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfbfcb481c8125039faa0c6b62aa8ab4
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c8bdf3f237ded64a1f2c71ce58420fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4271b3c81c66be9b2d472eb848d3f542
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


class TestPrimitiveOp_c9b7d2853cf5f0d7ada56f57b578fad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4acda443423e75d640e40df381552696
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9ce76a568acce4da2b7193f396170f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0970f9570a05caf47ec1e0978456dbd5
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed122f7580e4b47209392336ba1e8f75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376d3c9930ad9c649ff264780f663ad0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0dc15e887e3fdf73d6921363e78c27db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc
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


class TestPrimitiveOp_13f10ad294b157d3cc202b77768c476f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376d3c9930ad9c649ff264780f663ad0
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa356ad48d7f408cbb260a61cdf77801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_008c7df90bfafc172f40d98c1346fb09
    def get_inputs(self):
        return [
            paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4075a1f8b2c5c1043dbe9b83fa4a5181(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e3fed05d5dbcacb0351ff9705267a1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_205849a805b3efb81ae1bc1f18d7af67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c28418a8577a881b8538277157a41f99
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


class TestPrimitiveOp_8467725fab67eba36fc4bebcb37b5036(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_607dbf3164b7f41fb31cb30558fb6210
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d0b028c566106547d9e6b13993a0595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eed9d2e8f21b9208ba2629ea01774e22
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_afaa72cf83a0a46b2f36dcbb0ca23156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3923ecae0c120ba8da8690300f1a01fa
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b75c5e51a07f85e5c57ffbd0c6e5683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376d3c9930ad9c649ff264780f663ad0
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1387e5f822eb300f9d66201e79e8d255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_008c7df90bfafc172f40d98c1346fb09
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


class TestPrimitiveOp_d69822ba33ec1fcb7775f72d73dd4bf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfbfcb481c8125039faa0c6b62aa8ab4
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2429a80d59c675f317d107637670155f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4271b3c81c66be9b2d472eb848d3f542
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


class TestPrimitiveOp_e52b9ae5d6751c3f21daa9e63de633b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e52b9ae5d6751c3f21daa9e63de633b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e52b9ae5d6751c3f21daa9e63de633b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4c9d356394ac4328973e7a18ef21176b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f7142f8773d0a4f7c18cdaa96d3551b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c9d356394ac4328973e7a18ef21176b
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9b7d2853cf5f0d7ada56f57b578fad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4acda443423e75d640e40df381552696
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9ce76a568acce4da2b7193f396170f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0970f9570a05caf47ec1e0978456dbd5
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


class TestPrimitiveOp_e52b9ae5d6751c3f21daa9e63de633b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0dc15e887e3fdf73d6921363e78c27db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0dc15e887e3fdf73d6921363e78c27db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0dc15e887e3fdf73d6921363e78c27db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30f1abd78ed79b02d92e7c92a33dea4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c9d356394ac4328973e7a18ef21176b
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b75c5e51a07f85e5c57ffbd0c6e5683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376d3c9930ad9c649ff264780f663ad0
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1387e5f822eb300f9d66201e79e8d255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_008c7df90bfafc172f40d98c1346fb09
    def get_inputs(self):
        return [
            paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_21f924440324d316f35b914b2c862000(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 15, 25], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abdcb762d9af7e8f01ea00cb3feb0517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21f924440324d316f35b914b2c862000
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13f10ad294b157d3cc202b77768c476f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376d3c9930ad9c649ff264780f663ad0
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa356ad48d7f408cbb260a61cdf77801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_008c7df90bfafc172f40d98c1346fb09
    def get_inputs(self):
        return [
            paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0467a1e3b7727fd6d955583ed7600172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b6088acdf6636e5ec7167a352dab46
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


class TestPrimitiveOp_d69822ba33ec1fcb7775f72d73dd4bf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfbfcb481c8125039faa0c6b62aa8ab4
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2429a80d59c675f317d107637670155f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4271b3c81c66be9b2d472eb848d3f542
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


class PrimitiveOp_0102443f3b4eea96d86f62ce8a020c66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a47b32124d9caee0e2bdd899b17dc17f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0102443f3b4eea96d86f62ce8a020c66
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


class TestPrimitiveOp_33aa3e7ff071a88934354e8819ca1463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b6088acdf6636e5ec7167a352dab46
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad7e3507d7a800d489c62b1cc83e2b8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfa556999fc09aec69573a2f0f52ead
    def get_inputs(self):
        return [
            paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4e6cb7628b0a7cd95df7517b1c991f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b6088acdf6636e5ec7167a352dab46
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_023c7bfe4862c893333d1aa29c90792b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfa556999fc09aec69573a2f0f52ead
    def get_inputs(self):
        return [
            paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d8d9aeea514364b9ab4a1572ce802ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3c51b83cc67661be238cfb3cf825e4e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_41e68046e97c364fd1a6c722ad978a80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3923ecae0c120ba8da8690300f1a01fa
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


class TestPrimitiveOp_c9a4b7d1690543c443e4322f9de7ea34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0102443f3b4eea96d86f62ce8a020c66
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db99b1a55eb891492603967e7f7083ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4acda443423e75d640e40df381552696
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7f8fc7cc3e2b3447ddc64e91831e42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0970f9570a05caf47ec1e0978456dbd5
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a0a0315c764f65078c6ceecafd64015f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b6088acdf6636e5ec7167a352dab46
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


class TestPrimitiveOp_d5f82fc93f9f6a74507888625d749e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfbfcb481c8125039faa0c6b62aa8ab4
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c8bdf3f237ded64a1f2c71ce58420fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4271b3c81c66be9b2d472eb848d3f542
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


class TestPrimitiveOp_fc8b2187a220e3d05d12099acdbe4e80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be452c6a4d618ac942a42f5d1332af27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc
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


class TestPrimitiveOp_69b1fbb126d1d4c1941229af6340d7fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b75c5e51a07f85e5c57ffbd0c6e5683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376d3c9930ad9c649ff264780f663ad0
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1387e5f822eb300f9d66201e79e8d255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_008c7df90bfafc172f40d98c1346fb09
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


class TestPrimitiveOp_b0888f3b4d5bcbecbb2f56e6840a01c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376d3c9930ad9c649ff264780f663ad0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0dc15e887e3fdf73d6921363e78c27db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4637ce3b194f5b02d0bdf3b665dfdc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c801be0fed4762f1878f79764813c743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3923ecae0c120ba8da8690300f1a01fa
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


class TestPrimitiveOp_00c34176cd60be4e31bee07260a116a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b6088acdf6636e5ec7167a352dab46
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


class TestPrimitiveOp_d69822ba33ec1fcb7775f72d73dd4bf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfbfcb481c8125039faa0c6b62aa8ab4
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2429a80d59c675f317d107637670155f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4271b3c81c66be9b2d472eb848d3f542
    def get_inputs(self):
        return [
            paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()