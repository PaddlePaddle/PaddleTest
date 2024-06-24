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



class PrimitiveOp_f79edaa32ac6c3071c3fd00dce083d79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29eea5cc5f8ac1441add15c9ae7c042f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79edaa32ac6c3071c3fd00dce083d79
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_84793998d56a5b43308d58069fd7187d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58fad75e452dee2cc5c594046d460648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84793998d56a5b43308d58069fd7187d
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_331e94bee2bd309814f937015f72199a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21f695c7bac04e3be9dcf20b1bbb6af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_331e94bee2bd309814f937015f72199a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5e2ab8415cdd5e6e19c4c55b98d4b1b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c17a3f4a55ad6e95a6f4ee66e017e71d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e2ab8415cdd5e6e19c4c55b98d4b1b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_64b3612fb370de353a1b268c3b821236(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_173a788c7dcc2e2fae557fbff395174a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64b3612fb370de353a1b268c3b821236
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c5ada77aea41c1b14fb8d9e874274ab8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4cdd16c01bd63b0b75610cf0280e129f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ada77aea41c1b14fb8d9e874274ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0d01762121dbb8c04d505ae04a4ae0a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9f0092946410c5b46b82831d1f41c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d01762121dbb8c04d505ae04a4ae0a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_87526143ba591b5e347947beca130c76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ea1b1f9f24b2cd69b4b3f5ae2922ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87526143ba591b5e347947beca130c76
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e181865e0ba9f8a0e82bd65bd8272122(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_daa3c17d2bdad7e3ef413762a3195594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e181865e0ba9f8a0e82bd65bd8272122
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ac586694e388164438a8f3fe885eece7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76186274a8f7c7d0e7d5c3152ccec758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac586694e388164438a8f3fe885eece7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_74a9c3274e82fd70e05b97a97c64f676(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de1046a18a5f5319903d1e2b075a1ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74a9c3274e82fd70e05b97a97c64f676
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c2ffdc91577917ee4518ce373991df84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c710e32bdedbea78331a048c69e2c54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2ffdc91577917ee4518ce373991df84
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6a06100d93af8c10246d4d9b29f0be6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d67683acc6e169e88087ec5fa7b63001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a06100d93af8c10246d4d9b29f0be6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de1046a18a5f5319903d1e2b075a1ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74a9c3274e82fd70e05b97a97c64f676
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76186274a8f7c7d0e7d5c3152ccec758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac586694e388164438a8f3fe885eece7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c177c5e1c474f3174532ce773e1de32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6e935281a74f84740567baf06dd39234(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a18d717a51e9ed18b7a6bb30f136b22c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e935281a74f84740567baf06dd39234
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.217710256576538]], [[1.8326380252838135]], [[1.2897367477416992]], [[1.870718002319336]], [[1.794393539428711]], [[1.2489148378372192]], [[1.2155715227127075]], [[1.6209197044372559]], [[2.3135547637939453]], [[1.7274829149246216]], [[1.4666361808776855]], [[1.2465654611587524]], [[1.3598618507385254]], [[0.43374720215797424]], [[1.5060817003250122]], [[1.7152764797210693]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6c53ad76abc8e8d50f55fee93f7fb47c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 44, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18e88055cf1f13d94f9bb7715e16ac44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c53ad76abc8e8d50f55fee93f7fb47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c17a3f4a55ad6e95a6f4ee66e017e71d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e2ab8415cdd5e6e19c4c55b98d4b1b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de1046a18a5f5319903d1e2b075a1ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74a9c3274e82fd70e05b97a97c64f676
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29eea5cc5f8ac1441add15c9ae7c042f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79edaa32ac6c3071c3fd00dce083d79
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9f0092946410c5b46b82831d1f41c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d01762121dbb8c04d505ae04a4ae0a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cdd16c01bd63b0b75610cf0280e129f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ada77aea41c1b14fb8d9e874274ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e98a6057ab8ac29e7554ebf80879f084(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab77adfa52800ffaedacc5af48e4900f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e98a6057ab8ac29e7554ebf80879f084
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cdd16c01bd63b0b75610cf0280e129f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ada77aea41c1b14fb8d9e874274ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0b02b820ca3845b450ccc06fc185d037(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3244ea5ab118ad2b2c431328f2879520(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b02b820ca3845b450ccc06fc185d037
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29eea5cc5f8ac1441add15c9ae7c042f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79edaa32ac6c3071c3fd00dce083d79
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_daa3c17d2bdad7e3ef413762a3195594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e181865e0ba9f8a0e82bd65bd8272122
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f13ffc13df02b96e0c965046751b95cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_132a9c56c5542f70b08b59da10d84f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f13ffc13df02b96e0c965046751b95cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ef4004d2bfd110446d0cd9fc2342a3ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a1e5809c8e2ff4f705bc947bd646a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4004d2bfd110446d0cd9fc2342a3ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de1046a18a5f5319903d1e2b075a1ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74a9c3274e82fd70e05b97a97c64f676
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cdd16c01bd63b0b75610cf0280e129f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ada77aea41c1b14fb8d9e874274ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ec9d38a776abf5ba1e4f243247f6880a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_873fef0881c4350ef309136433df873a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec9d38a776abf5ba1e4f243247f6880a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76186274a8f7c7d0e7d5c3152ccec758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac586694e388164438a8f3fe885eece7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e8e48567d534d700c51eb2a8fb93f95d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d034535ffb2022244834dfbb95b59ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e48567d534d700c51eb2a8fb93f95d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1ca16c2429fb25bce935f97b484db58f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2b45a2bfdae3d6d6790aafbac992415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca16c2429fb25bce935f97b484db58f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cdd16c01bd63b0b75610cf0280e129f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ada77aea41c1b14fb8d9e874274ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_132a9c56c5542f70b08b59da10d84f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f13ffc13df02b96e0c965046751b95cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6e768ecbca6e75bbe2c933dabcadaee2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92aa49fb7acb301fcf26f80fe9190688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e768ecbca6e75bbe2c933dabcadaee2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cdd16c01bd63b0b75610cf0280e129f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ada77aea41c1b14fb8d9e874274ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ea1b1f9f24b2cd69b4b3f5ae2922ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87526143ba591b5e347947beca130c76
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d83fa2beee3b09bea615452645e40331(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e80fa63a9186d94ea3141c9655c5d84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d83fa2beee3b09bea615452645e40331
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_173a788c7dcc2e2fae557fbff395174a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64b3612fb370de353a1b268c3b821236
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58fad75e452dee2cc5c594046d460648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84793998d56a5b43308d58069fd7187d
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e80fa63a9186d94ea3141c9655c5d84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d83fa2beee3b09bea615452645e40331
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76186274a8f7c7d0e7d5c3152ccec758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac586694e388164438a8f3fe885eece7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cdd16c01bd63b0b75610cf0280e129f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ada77aea41c1b14fb8d9e874274ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d034535ffb2022244834dfbb95b59ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e48567d534d700c51eb2a8fb93f95d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d67683acc6e169e88087ec5fa7b63001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a06100d93af8c10246d4d9b29f0be6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ea1b1f9f24b2cd69b4b3f5ae2922ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87526143ba591b5e347947beca130c76
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f695c7bac04e3be9dcf20b1bbb6af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_331e94bee2bd309814f937015f72199a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d67683acc6e169e88087ec5fa7b63001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a06100d93af8c10246d4d9b29f0be6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_daa3c17d2bdad7e3ef413762a3195594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e181865e0ba9f8a0e82bd65bd8272122
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3244ea5ab118ad2b2c431328f2879520(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b02b820ca3845b450ccc06fc185d037
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_596ffab083ab7f492fd2c4eb0df867f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe75e5bdc21000adfeda2add9b20bd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ffab083ab7f492fd2c4eb0df867f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76186274a8f7c7d0e7d5c3152ccec758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac586694e388164438a8f3fe885eece7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_173a788c7dcc2e2fae557fbff395174a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64b3612fb370de353a1b268c3b821236
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cdd16c01bd63b0b75610cf0280e129f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ada77aea41c1b14fb8d9e874274ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_eb3a6d793e39f6ca50f69b7a3168f3c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c35f382873cfe2a3f3972e4c53784028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb3a6d793e39f6ca50f69b7a3168f3c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f695c7bac04e3be9dcf20b1bbb6af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_331e94bee2bd309814f937015f72199a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76186274a8f7c7d0e7d5c3152ccec758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac586694e388164438a8f3fe885eece7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d67683acc6e169e88087ec5fa7b63001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a06100d93af8c10246d4d9b29f0be6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c094cc016ba7897b5967af20d17d184(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e81667b41779817dc3a50028e5c1305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c094cc016ba7897b5967af20d17d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c17a3f4a55ad6e95a6f4ee66e017e71d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e2ab8415cdd5e6e19c4c55b98d4b1b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_db942f0524d921fd975140954fa48356(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80de570465d0e79e9092e1c135257cd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db942f0524d921fd975140954fa48356
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92aa49fb7acb301fcf26f80fe9190688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e768ecbca6e75bbe2c933dabcadaee2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d67683acc6e169e88087ec5fa7b63001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a06100d93af8c10246d4d9b29f0be6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f695c7bac04e3be9dcf20b1bbb6af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_331e94bee2bd309814f937015f72199a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0508f050264367f104b99b00f51712d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e935281a74f84740567baf06dd39234
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4607528448104858]], [[1.2040683031082153]], [[1.2330397367477417]], [[1.8749967813491821]], [[1.7495776414871216]], [[1.6248310804367065]], [[1.688737392425537]], [[2.427034378051758]], [[1.5513508319854736]], [[1.9906879663467407]], [[1.7836780548095703]], [[1.5312771797180176]], [[1.893558144569397]], [[1.569359540939331]], [[1.3054555654525757]], [[1.3310296535491943]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e81667b41779817dc3a50028e5c1305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c094cc016ba7897b5967af20d17d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_873fef0881c4350ef309136433df873a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec9d38a776abf5ba1e4f243247f6880a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f082828ca2b281c978c788786e552aa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 400, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a30c450c584c408def3e4b2c5f83a96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f082828ca2b281c978c788786e552aa1
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d034535ffb2022244834dfbb95b59ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e48567d534d700c51eb2a8fb93f95d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_343b9d8f8f263e262c01416b8dcb65d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f45e6621b7efca7b8f71347617003f43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_343b9d8f8f263e262c01416b8dcb65d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29eea5cc5f8ac1441add15c9ae7c042f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79edaa32ac6c3071c3fd00dce083d79
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cdd16c01bd63b0b75610cf0280e129f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ada77aea41c1b14fb8d9e874274ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2b45a2bfdae3d6d6790aafbac992415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca16c2429fb25bce935f97b484db58f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b44e725aa1d8afd6dc78b773b81b5a2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01cf25ad4f7d286bcfc2715bc243adee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b44e725aa1d8afd6dc78b773b81b5a2f
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f695c7bac04e3be9dcf20b1bbb6af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_331e94bee2bd309814f937015f72199a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2b45a2bfdae3d6d6790aafbac992415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca16c2429fb25bce935f97b484db58f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18e88055cf1f13d94f9bb7715e16ac44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c53ad76abc8e8d50f55fee93f7fb47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92aa49fb7acb301fcf26f80fe9190688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e768ecbca6e75bbe2c933dabcadaee2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a30c450c584c408def3e4b2c5f83a96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f082828ca2b281c978c788786e552aa1
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e610a5e2542707be4f4a088019ddc6a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 56, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1482c60edc833e84d5217b2c1d69c63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e610a5e2542707be4f4a088019ddc6a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76186274a8f7c7d0e7d5c3152ccec758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac586694e388164438a8f3fe885eece7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29eea5cc5f8ac1441add15c9ae7c042f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79edaa32ac6c3071c3fd00dce083d79
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe75e5bdc21000adfeda2add9b20bd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ffab083ab7f492fd2c4eb0df867f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c17a3f4a55ad6e95a6f4ee66e017e71d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e2ab8415cdd5e6e19c4c55b98d4b1b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2b45a2bfdae3d6d6790aafbac992415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca16c2429fb25bce935f97b484db58f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76186274a8f7c7d0e7d5c3152ccec758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac586694e388164438a8f3fe885eece7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_85630df1db5f563fe79fea251a2c71c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_64620bb4de26d39cab33f7ab9d4a506a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85630df1db5f563fe79fea251a2c71c2
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64620bb4de26d39cab33f7ab9d4a506a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85630df1db5f563fe79fea251a2c71c2
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64620bb4de26d39cab33f7ab9d4a506a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85630df1db5f563fe79fea251a2c71c2
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64620bb4de26d39cab33f7ab9d4a506a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85630df1db5f563fe79fea251a2c71c2
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4cf780d550f2b797fc579673ce038090(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4095d1f0375a3b22d0609958e35b936a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cf780d550f2b797fc579673ce038090
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4095d1f0375a3b22d0609958e35b936a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cf780d550f2b797fc579673ce038090
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4095d1f0375a3b22d0609958e35b936a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cf780d550f2b797fc579673ce038090
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4095d1f0375a3b22d0609958e35b936a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cf780d550f2b797fc579673ce038090
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92aa49fb7acb301fcf26f80fe9190688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e768ecbca6e75bbe2c933dabcadaee2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9f0092946410c5b46b82831d1f41c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d01762121dbb8c04d505ae04a4ae0a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9f0092946410c5b46b82831d1f41c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d01762121dbb8c04d505ae04a4ae0a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9f0092946410c5b46b82831d1f41c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d01762121dbb8c04d505ae04a4ae0a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8d772694a0bca7288e934d5eaac3ab88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 200, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_273d02485caea706e14842369b9a9ecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d772694a0bca7288e934d5eaac3ab88
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e81667b41779817dc3a50028e5c1305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c094cc016ba7897b5967af20d17d184
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76186274a8f7c7d0e7d5c3152ccec758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac586694e388164438a8f3fe885eece7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9f0092946410c5b46b82831d1f41c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d01762121dbb8c04d505ae04a4ae0a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f695c7bac04e3be9dcf20b1bbb6af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_331e94bee2bd309814f937015f72199a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f695c7bac04e3be9dcf20b1bbb6af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_331e94bee2bd309814f937015f72199a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a1e5809c8e2ff4f705bc947bd646a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4004d2bfd110446d0cd9fc2342a3ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d034535ffb2022244834dfbb95b59ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e48567d534d700c51eb2a8fb93f95d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b25eb871be0a67ba974dde73fe3d430f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ac894f047d9b160d6a876a60b30ff0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b25eb871be0a67ba974dde73fe3d430f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76186274a8f7c7d0e7d5c3152ccec758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac586694e388164438a8f3fe885eece7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_173a788c7dcc2e2fae557fbff395174a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64b3612fb370de353a1b268c3b821236
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c35f382873cfe2a3f3972e4c53784028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb3a6d793e39f6ca50f69b7a3168f3c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f695c7bac04e3be9dcf20b1bbb6af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_331e94bee2bd309814f937015f72199a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ea1b1f9f24b2cd69b4b3f5ae2922ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87526143ba591b5e347947beca130c76
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_173a788c7dcc2e2fae557fbff395174a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64b3612fb370de353a1b268c3b821236
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_273d02485caea706e14842369b9a9ecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d772694a0bca7288e934d5eaac3ab88
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_132a9c56c5542f70b08b59da10d84f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f13ffc13df02b96e0c965046751b95cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92aa49fb7acb301fcf26f80fe9190688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e768ecbca6e75bbe2c933dabcadaee2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76186274a8f7c7d0e7d5c3152ccec758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac586694e388164438a8f3fe885eece7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a2c7334268a375ef0b05caa831cc17a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78a3966e1989ab6b88d70c1396144d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c7334268a375ef0b05caa831cc17a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76186274a8f7c7d0e7d5c3152ccec758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac586694e388164438a8f3fe885eece7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92aa49fb7acb301fcf26f80fe9190688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e768ecbca6e75bbe2c933dabcadaee2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d67683acc6e169e88087ec5fa7b63001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6a06100d93af8c10246d4d9b29f0be6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c35f382873cfe2a3f3972e4c53784028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb3a6d793e39f6ca50f69b7a3168f3c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cdd16c01bd63b0b75610cf0280e129f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ada77aea41c1b14fb8d9e874274ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cdd16c01bd63b0b75610cf0280e129f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ada77aea41c1b14fb8d9e874274ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d034535ffb2022244834dfbb95b59ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e48567d534d700c51eb2a8fb93f95d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9f0092946410c5b46b82831d1f41c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d01762121dbb8c04d505ae04a4ae0a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c17a3f4a55ad6e95a6f4ee66e017e71d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e2ab8415cdd5e6e19c4c55b98d4b1b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f695c7bac04e3be9dcf20b1bbb6af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_331e94bee2bd309814f937015f72199a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab77adfa52800ffaedacc5af48e4900f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e98a6057ab8ac29e7554ebf80879f084
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3244ea5ab118ad2b2c431328f2879520(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b02b820ca3845b450ccc06fc185d037
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b456627706ce1b62e77a420ec9ad3c45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4291af32fe3e07695d71e77b7ce91e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b456627706ce1b62e77a420ec9ad3c45
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01cf25ad4f7d286bcfc2715bc243adee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b44e725aa1d8afd6dc78b773b81b5a2f
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe75e5bdc21000adfeda2add9b20bd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ffab083ab7f492fd2c4eb0df867f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76186274a8f7c7d0e7d5c3152ccec758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac586694e388164438a8f3fe885eece7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1482c60edc833e84d5217b2c1d69c63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e610a5e2542707be4f4a088019ddc6a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e80fa63a9186d94ea3141c9655c5d84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d83fa2beee3b09bea615452645e40331
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2b45a2bfdae3d6d6790aafbac992415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca16c2429fb25bce935f97b484db58f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_132a9c56c5542f70b08b59da10d84f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f13ffc13df02b96e0c965046751b95cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9f0092946410c5b46b82831d1f41c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d01762121dbb8c04d505ae04a4ae0a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de1046a18a5f5319903d1e2b075a1ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74a9c3274e82fd70e05b97a97c64f676
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c17a3f4a55ad6e95a6f4ee66e017e71d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e2ab8415cdd5e6e19c4c55b98d4b1b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4291af32fe3e07695d71e77b7ce91e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b456627706ce1b62e77a420ec9ad3c45
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ea1b1f9f24b2cd69b4b3f5ae2922ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87526143ba591b5e347947beca130c76
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f45e6621b7efca7b8f71347617003f43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_343b9d8f8f263e262c01416b8dcb65d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f00f04fb884c3d958e4e70992be8bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a1c1a79597dfe808eaa0d0049ddaa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_daa3c17d2bdad7e3ef413762a3195594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e181865e0ba9f8a0e82bd65bd8272122
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_273d02485caea706e14842369b9a9ecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d772694a0bca7288e934d5eaac3ab88
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a30c450c584c408def3e4b2c5f83a96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f082828ca2b281c978c788786e552aa1
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a1e5809c8e2ff4f705bc947bd646a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef4004d2bfd110446d0cd9fc2342a3ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f695c7bac04e3be9dcf20b1bbb6af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_331e94bee2bd309814f937015f72199a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92aa49fb7acb301fcf26f80fe9190688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e768ecbca6e75bbe2c933dabcadaee2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c17a3f4a55ad6e95a6f4ee66e017e71d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e2ab8415cdd5e6e19c4c55b98d4b1b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_132a9c56c5542f70b08b59da10d84f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f13ffc13df02b96e0c965046751b95cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3244ea5ab118ad2b2c431328f2879520(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b02b820ca3845b450ccc06fc185d037
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c17a3f4a55ad6e95a6f4ee66e017e71d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e2ab8415cdd5e6e19c4c55b98d4b1b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92aa49fb7acb301fcf26f80fe9190688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e768ecbca6e75bbe2c933dabcadaee2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc505fdcaf85a2fe10f0aa1f464cf90e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c177c5e1c474f3174532ce773e1de32
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c710e32bdedbea78331a048c69e2c54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2ffdc91577917ee4518ce373991df84
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c710e32bdedbea78331a048c69e2c54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2ffdc91577917ee4518ce373991df84
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c710e32bdedbea78331a048c69e2c54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2ffdc91577917ee4518ce373991df84
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c710e32bdedbea78331a048c69e2c54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2ffdc91577917ee4518ce373991df84
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cf8dfebaba8dae40958068dc41fcad73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a0dc44dcef1a1e8283bcb878f797f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf8dfebaba8dae40958068dc41fcad73
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[61632.31640625]], [[52611.7890625]], [[60782.27734375]], [[41501.59765625]], [[36700.984375]], [[33332.984375]], [[49396.5546875]], [[49001.73828125]], [[46942.0625]], [[55142.59375]], [[26716.734375]], [[40395.26953125]], [[53856.0625]], [[44029.78125]], [[39522.9140625]], [[43628.35546875]], [[53016.65625]], [[48293.76171875]], [[59256.7109375]], [[35550.72265625]], [[60129.203125]], [[40471.23046875]], [[40736.60546875]], [[35122.7265625]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_be8028e0a0aa418b23d465d990a5c869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf8dfebaba8dae40958068dc41fcad73
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[70247.6171875]], [[55049.859375]], [[57602.5234375]], [[41331.9609375]], [[34158.9765625]], [[56500.3828125]], [[65477.69140625]], [[60592.91796875]], [[59113.06640625]], [[60905.34765625]], [[53484.515625]], [[57595.3828125]], [[70439.3984375]], [[61898.45703125]], [[46357.421875]], [[76779.5078125]], [[54456.20703125]], [[43128.85546875]], [[60067.00390625]], [[77821.9453125]], [[59378.29296875]], [[68799.8359375]], [[66236.2109375]], [[45041.50390625]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_2dbfd68e875b15c535e1cba284859351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf8dfebaba8dae40958068dc41fcad73
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[47563.6484375]], [[33220.03515625]], [[28940.765625]], [[79870.828125]], [[38200.72265625]], [[59941.8359375]], [[48318.53515625]], [[68341.9453125]], [[51844.328125]], [[55875.546875]], [[72798.8671875]], [[74809.5546875]], [[76797.0859375]], [[56505.67578125]], [[33038.90234375]], [[52546.7421875]], [[72962.609375]], [[80556.4921875]], [[55447.8203125]], [[73883.5]], [[79402.0078125]], [[43280.1953125]], [[45789.3828125]], [[47405.984375]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_3a9cdc7ea087ea30734b94cc24dbdef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf8dfebaba8dae40958068dc41fcad73
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[79026.1015625]], [[49374.56640625]], [[61700.61328125]], [[68773.453125]], [[38092.93359375]], [[35693.65234375]], [[54690.87109375]], [[84150.1484375]], [[72123.578125]], [[64649.27734375]], [[84920.8046875]], [[54669.828125]], [[60167.1875]], [[56453.55859375]], [[24903.0703125]], [[51346.796875]], [[46619.66796875]], [[49899.26953125]], [[48152.5859375]], [[37075.6953125]], [[54901.3125]], [[71285.0078125]], [[57870.62109375]], [[74277.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_de1046a18a5f5319903d1e2b075a1ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74a9c3274e82fd70e05b97a97c64f676
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ac894f047d9b160d6a876a60b30ff0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b25eb871be0a67ba974dde73fe3d430f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c710e32bdedbea78331a048c69e2c54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2ffdc91577917ee4518ce373991df84
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ea1b1f9f24b2cd69b4b3f5ae2922ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87526143ba591b5e347947beca130c76
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cdd16c01bd63b0b75610cf0280e129f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ada77aea41c1b14fb8d9e874274ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()