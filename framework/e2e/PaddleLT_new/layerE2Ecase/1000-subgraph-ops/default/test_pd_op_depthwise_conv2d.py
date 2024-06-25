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



class PrimitiveOp_597de61270330effc9bcda849a60d2af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ec39002adbcaad85f1b04e7b1b09e15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_597de61270330effc9bcda849a60d2af
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_16ef387bd9145ae1a15f755c77f1011f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_264ff44dfd29f5fc7b1278d836bb5e5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16ef387bd9145ae1a15f755c77f1011f
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6d59a0a33c7a6c7d401706e51813fe7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 48, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 1, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d345e5f04362703cf6337642e86e3d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d59a0a33c7a6c7d401706e51813fe7d
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9006be2c9379017591e4c939feb23859(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_240ba7ba8e3a7564562854fc29424b12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9006be2c9379017591e4c939feb23859
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_240ba7ba8e3a7564562854fc29424b12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9006be2c9379017591e4c939feb23859
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_37b618684340be2a39f3f5cc19dee7e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [2, 2], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_efc4cde6a02ad9ff49eea35dafdd0f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37b618684340be2a39f3f5cc19dee7e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f8be2674fff93a7d515edbc4013b16b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [3, 3], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_085220dea1dcb00d6ef4c60b4b0979bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8be2674fff93a7d515edbc4013b16b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8dccc1d519d2c746dbf02e8d273f953b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 1280, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1280, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1280, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f345a4e5e1dc5278f4d787a4fb5b457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dccc1d519d2c746dbf02e8d273f953b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fbed2532e03efc6371254b1643b7ac20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37a2a8bb599f3801d1e306278e37bd79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbed2532e03efc6371254b1643b7ac20
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b8a5cfb4827ae3a5a67e1b5949f2e213(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [2, 2], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_30ff23623f834726e6e9a4793bf6a765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8a5cfb4827ae3a5a67e1b5949f2e213
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0966e0ea1fc18554b36cb79a174d09d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [3, 3], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5be7d015c91bca1b069056631b4bd5c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0966e0ea1fc18554b36cb79a174d09d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_312d5b77d8c07c85ab524757cb77be44(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [4, 4], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8156960a2eb0e426603a7343143df31d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_312d5b77d8c07c85ab524757cb77be44
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_70b642ba671a48c5cc366e22534fc7de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92409a76153ebe9cfb81c2c33f5edd56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70b642ba671a48c5cc366e22534fc7de
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5ee875c189087fadfe1dad11d7b775bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e10a6a19fdf2dbb98114b04ccbd229c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ee875c189087fadfe1dad11d7b775bb
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 1, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4d5f0707980d4b2eeaa4b4173e52074c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 160, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cd8f1e0d689c1093a3808dfc796f83d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d5f0707980d4b2eeaa4b4173e52074c
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 1, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_76b20ef6a28154e1d11cfce3d791c07e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 192, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed4e1ddaf507bc5fcafffbf53fdeb14e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76b20ef6a28154e1d11cfce3d791c07e
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f2853f91befd0c6757fdd392d223f9fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 384, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a92500eb4eddcbda5c6416a21216ae4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2853f91befd0c6757fdd392d223f9fe
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_247a10507da7ba145e0ace2a4943a5e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fcbd4513a171d82a1f36aad44bdb4f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_247a10507da7ba145e0ace2a4943a5e0
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_427f7330f0057346239f5f1b3b78e966(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 80, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc1bc3ba6458a5b3c45856e4e5af52fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_427f7330f0057346239f5f1b3b78e966
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 1, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_918907168f6c1a5429e4c039fabbbc8c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 80, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 1, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_887fe260d8b9c0207053359c513f578c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_918907168f6c1a5429e4c039fabbbc8c
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 1, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aac1d37b91298291234400c243bf10f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 300, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6659c2a24abbdfc37172177d898c2462(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aac1d37b91298291234400c243bf10f2
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_217e9aebd7a68cc7e5f1e153ebf00e91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 300, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e152ca6b1e255ce7aae7cc3fac1dc1e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_217e9aebd7a68cc7e5f1e153ebf00e91
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 1, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6912dff83c92b3c9f3500d838d4a3c88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 300, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 1, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7fd00ac8e71ad5181b551f3883e9f4a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6912dff83c92b3c9f3500d838d4a3c88
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 1, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5469167058c2d651677e4b0902bcf4f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 300, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 1, 9, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b5d52b0b3e0ddbe366e2bbc493bfd31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5469167058c2d651677e4b0902bcf4f8
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 1, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_43e4e78493fb8531eed0507b7c5c9db6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61a73141456fbbd9a3340af5543ba0db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43e4e78493fb8531eed0507b7c5c9db6
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cc6b6eaccf3aed9bbc8c95070d2cafdb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 90, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[90, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_251383338c1500074b7a583141a2f830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc6b6eaccf3aed9bbc8c95070d2cafdb
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([90, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_82ff7534e65968be7628ae3321d9fbb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 90, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[90, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7320c177698a4d737b660c27a22e8f87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82ff7534e65968be7628ae3321d9fbb4
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([90, 1, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e257ae05638ee23b6e30e9ee4cc4fc71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 90, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[90, 1, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff88c300c684e90f519458ab17a28cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e257ae05638ee23b6e30e9ee4cc4fc71
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([90, 1, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b708458609ff41467cf072bbb6a18d83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 90, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[90, 1, 9, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04b75deacb479836f54c384c9e783326(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b708458609ff41467cf072bbb6a18d83
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([90, 1, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_de93e8e4bc30856cf22db1d579c0bd51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 384, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11a8d8155b12bc6af6b94b9c988bad16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de93e8e4bc30856cf22db1d579c0bd51
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a071a55953f88c05fbf418b982b21168(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e203ad9976a1373a1545d7751510451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a071a55953f88c05fbf418b982b21168
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_45c178378dbf762eeada87489ec49b67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc5f02146261d790439feee5f0f1e1e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45c178378dbf762eeada87489ec49b67
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 1, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ae42c50c8988f810c2c42a8b3ab9206f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 1, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36a145c9eadbda0358139b948115d129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae42c50c8988f810c2c42a8b3ab9206f
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 1, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f52ce0a7a44e90fee4b975e8d38c3a54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [4, 4], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 1, 9, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d86adb5ed0c05ca76a108623c5de337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ce0a7a44e90fee4b975e8d38c3a54
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 1, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa8133f5f2870424e4ba46ec2f9fcb64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [5, 5], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 1, 11, 11], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_949bfbd4dc1091b0404f2919bfdadc38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa8133f5f2870424e4ba46ec2f9fcb64
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 1, 11, 11], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_71b0c0ba327986b861449c9c1dfcc377(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 768, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 768, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2121adbf00bb2f6115b4f5722dee390b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71b0c0ba327986b861449c9c1dfcc377
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31fd69823b0519fdd41ad347c249e518(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dccc1d519d2c746dbf02e8d273f953b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_40bb1c70d2f3d1cd07a5bc0f5089e299(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 768, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 768, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_051813746207560fa87a14046e2f4d86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40bb1c70d2f3d1cd07a5bc0f5089e299
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e746aaf353c23f1a0462699d3359203c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f33167b1fbeaae469b0e1703b31d38fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e746aaf353c23f1a0462699d3359203c
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b3fe94ee476440a3c2b71f2df29d6afd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [2, 2], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_223527641170e7258274e772aef8d82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3fe94ee476440a3c2b71f2df29d6afd
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5e14a9761590bfa6291b02495f4f7ef2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [3, 3], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4429fc9bf2824196f4e9c10626f8483d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e14a9761590bfa6291b02495f4f7ef2
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_587e61387073cbf263553b5d46e1bfda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [4, 4], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf98103dcad8f114602b8ca94452a665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_587e61387073cbf263553b5d46e1bfda
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0b582811a02e141ddeff01496c72f514(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b411c5f76e14755656b6ce747d2e5055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b582811a02e141ddeff01496c72f514
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_82b2ef0f09500927dfa13eb50772fc47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [2, 2], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_107a9c164069b97794acb12cb14dd26f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82b2ef0f09500927dfa13eb50772fc47
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8aa5bcdacfddd186cd31f8ee81862697(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [3, 3], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b09fed8530aef947f964ba24262eadd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aa5bcdacfddd186cd31f8ee81862697
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_533bb47dbdcb4f97dfc37811d4b074cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [4, 4], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8518f3b3f0af6915cba05c856178eb6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_533bb47dbdcb4f97dfc37811d4b074cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e220cd6b83d4c8d00824906f13c698ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 192, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_84c76947397a9aebeedb3217068d9197(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e220cd6b83d4c8d00824906f13c698ea
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_39468bba57ad36e83ff23cd53620a02c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 240, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2840a9f091f82ac50a9914a0eeda8c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39468bba57ad36e83ff23cd53620a02c
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bc57b382620ca16bf0a152567f93272d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 240, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7456cbf5b5b581d211dcd9455279b6e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc57b382620ca16bf0a152567f93272d
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 1, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9c99f9a1997e1e4d6201f1a26297ec6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_820ded4cef96de37c4e006d31fb96100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c99f9a1997e1e4d6201f1a26297ec6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()