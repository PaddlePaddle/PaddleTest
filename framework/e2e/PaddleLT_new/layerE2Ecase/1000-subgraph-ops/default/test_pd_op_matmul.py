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



class PrimitiveOp_61f36adf9261cb243ad6050979d45402(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_126c8c65c9a68de5bd61ca5aca715467(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61f36adf9261cb243ad6050979d45402
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_14a629e06483c2d204831e741fffc750(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f47590b2e9e8c2c9f318c11a03147a6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14a629e06483c2d204831e741fffc750
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4f27ab9b93e77c852a668b6fd1d024f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_214252e4f9ea802bf22705031cb55e3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f27ab9b93e77c852a668b6fd1d024f6
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ce5439a2ce2d3081c3549ab197b62a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 8, 8, 7, 7, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca9e81d781f681a2e686b07c549e6385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ce5439a2ce2d3081c3549ab197b62a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f6ae809cca6e6475c9470527188535d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[72, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37824d5eb28274571c1c64beead41aa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ae809cca6e6475c9470527188535d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb8fa6581e193a582194c63552ee3d74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[18, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61ca774673d39f79aa4200dbc3fc1550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb8fa6581e193a582194c63552ee3d74
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.826145172119141, 4.080577850341797, 4.445215225219727, 4.608267784118652, 4.19982385635376, 4.360614776611328, 3.9698872566223145, 4.988100051879883, 4.204438209533691, 4.753846645355225, 3.976109743118286, 4.695165157318115, 4.921513080596924, 4.550264835357666, 4.6359663009643555, 4.788970947265625, 4.956936359405518, 4.846409797668457]], dtype='float32').reshape([1, 18]),
            paddle.uniform([18, 72], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ea81763322d21cd10e9dccaa15ae2693(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_532cec7e7bdb8e619bc9f558f667080a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea81763322d21cd10e9dccaa15ae2693
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_afe4e8bf8bba039bdf4c6b915ea202dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4, 32, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b1b175c4577954f0a45311744110719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afe4e8bf8bba039bdf4c6b915ea202dd
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_94eef12a2b31cdbde24acd18cb974e2e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1826d1389292b374ecb255faa24f83fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94eef12a2b31cdbde24acd18cb974e2e
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_34c98fdb5c8c3816f9a7d51e9e8943bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b1b0517d917fb2a2def90556585a93f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34c98fdb5c8c3816f9a7d51e9e8943bc
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e7880d1e54db8d78b50628ddf9c55ce2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[92, 23], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2b9da9f9043d2e2e70c0bf30cf46fc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7880d1e54db8d78b50628ddf9c55ce2
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([92, 23], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4d0e230262a12d0ab1f652dcdb8e9d2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 23], dtype='float32'),
            paddle.static.InputSpec(shape=[23, 92], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c139e2019dd53ad836b5f77f6bb2f07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d0e230262a12d0ab1f652dcdb8e9d2d
    def get_inputs(self):
        return [
            paddle.to_tensor([[6.905711650848389, 6.423632621765137, 6.014829158782959, 6.478201389312744, 5.563326358795166, 5.671665191650391, 5.8199687004089355, 6.04303503036499, 6.119690895080566, 6.3701019287109375, 5.8362956047058105, 5.950503349304199, 6.3348917961120605, 6.629144668579102, 6.015385627746582, 6.078804969787598, 5.6398091316223145, 6.834695816040039, 6.461501121520996, 6.278164863586426, 5.707799911499023, 6.065580368041992, 5.976922988891602]], dtype='float32').reshape([1, 23]),
            paddle.uniform([23, 92], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_06625f1bd7809d4bd5b7c01f914d78ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0eeba47ec89598fc4c7aaa2f21e9a9a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06625f1bd7809d4bd5b7c01f914d78ac
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c9b2b09ab4307a715cab833c43bcff35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 198, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 3, 64, 198], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e465dde32ec3b4ae9c978299b8e93f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b2b09ab4307a715cab833c43bcff35
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([54, 3, 64, 198], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_04f35582cc04e281a68786b43457b3e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 3, 198, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d323f2ef48698599ef75055fe849f60a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04f35582cc04e281a68786b43457b3e8
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0c4fab6a958c2b9edd772574465764e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19b3bcf77a271884f522c2b139041f60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c4fab6a958c2b9edd772574465764e7
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2b21e3ae0b0556a504758b0ef5912cef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_810f6d196541b30b605548c39fbeb692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b21e3ae0b0556a504758b0ef5912cef
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_10754737f3209fd76fe18c6f5b939c25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f85ad29fbdf8d8317677ed5b5ba5820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10754737f3209fd76fe18c6f5b939c25
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2d69680c6b6b46b8dcadfa5b00d0541d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f1e493f899b080cd270244cd6bde4ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d69680c6b6b46b8dcadfa5b00d0541d
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c00f1dd70597e2c1e7ca20b05e107719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14a629e06483c2d204831e741fffc750
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_96288a25145efa7447b500a4ae2ae028(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e45d7d8133f242a266f1803e370b05c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96288a25145efa7447b500a4ae2ae028
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0f54080c1fc00ff17847cf1bfb64b0e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[960, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77507f66a8c8ed662965e2a84ae39e71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f54080c1fc00ff17847cf1bfb64b0e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_72b382779d871b23222374e793579269(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 960], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da80ede797e26054b88a9d5f3a301a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72b382779d871b23222374e793579269
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_76beb765e1b16d1c417b734b95f6c70d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2d06338149dca56122d3962f778295e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76beb765e1b16d1c417b734b95f6c70d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5e105e2bd3e16d933cf3aac2608b7b87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f013b855aa5c84b5bc15e85e5922bcf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e105e2bd3e16d933cf3aac2608b7b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_66fd5121f060d57cba8df01e0c075c33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[12544, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e1bd2b732ea8fec9a890d71a1af95a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66fd5121f060d57cba8df01e0c075c33
    def get_inputs(self):
        return [
            paddle.uniform([512, 12544], dtype='float32', min=0, max=0.5),
            paddle.uniform([12544, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8e374bd3daac99ba78bc8af729540392(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ddea5d1e5df7b1d8cfbc5d1655c7ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e374bd3daac99ba78bc8af729540392
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_801a1ac71b0650f60862a3ec37c444d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa764ba2bb934690e20cb08af12b1697(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_801a1ac71b0650f60862a3ec37c444d8
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_30b37d63558c53c0ec0e1dc9de8c9792(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c473c446a1110184aca29cddeb8ea77d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30b37d63558c53c0ec0e1dc9de8c9792
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ddfa228cfaacd7ffe628a2c2ae74c0ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 8, 8, 7, 7, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b11779a4b64bd91628cffd8efee5ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddfa228cfaacd7ffe628a2c2ae74c0ff
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_42a8bd5dd4855046b829f11f0b40f0e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0859cee343466b33fde3046674a5a1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42a8bd5dd4855046b829f11f0b40f0e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_37bf22e58c07c5e4eaa5c732c8ebb619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42a8bd5dd4855046b829f11f0b40f0e5
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b0d33df3f68ae41bbd94de1e59b80100(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe1fc53b790109336f05e7554fdc708b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0d33df3f68ae41bbd94de1e59b80100
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1e0e8ceda2f0a52cbe4b6f1d46f3bd5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a5b158c6824c39c53a2b896f86cd0cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e0e8ceda2f0a52cbe4b6f1d46f3bd5e
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ba7a7482e002a5b74974776b13034499(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b915625d9eec970a786e6eccb22e74b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba7a7482e002a5b74974776b13034499
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c24a26d6b17e8c38eec76c97cebf94e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_801a1ac71b0650f60862a3ec37c444d8
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32d5955a5e309d9c0deb162a25af7948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30b37d63558c53c0ec0e1dc9de8c9792
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ff288b9893decbec8411e64f9662434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d69680c6b6b46b8dcadfa5b00d0541d
    def get_inputs(self):
        return [
            paddle.uniform([11, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c24a26d6b17e8c38eec76c97cebf94e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_801a1ac71b0650f60862a3ec37c444d8
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32d5955a5e309d9c0deb162a25af7948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30b37d63558c53c0ec0e1dc9de8c9792
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_987aa13af058730fe4674c6d0e85eaca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60ce36f8d1b6eead74f9bc6deac9303c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_987aa13af058730fe4674c6d0e85eaca
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_089cb118a44d34edb3ac535724f982cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abfea4616096a0a9c09f84ca4506efd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_089cb118a44d34edb3ac535724f982cf
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9653d5ba560423ea1b1cf5b8319e1cd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea81763322d21cd10e9dccaa15ae2693
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4cf1fb083fd1679467c5fdcb80423be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afe4e8bf8bba039bdf4c6b915ea202dd
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7631475f03e0210f3f0400a399f07d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94eef12a2b31cdbde24acd18cb974e2e
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c463cc9d8038700faaf953d3782281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34c98fdb5c8c3816f9a7d51e9e8943bc
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e5b2858151b156d1ede2fcff61496bed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_544aed36126dbedba3b2277e36c9adba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b2858151b156d1ede2fcff61496bed
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_83f54cfd2cbb54040ed22e3633bea1ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_affc6d95714a7308c51889ed1dd49f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83f54cfd2cbb54040ed22e3633bea1ff
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a8d0b2fbe162f672c6bb64964189e9ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 4, 4, 7, 7, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_114675e8dec6ea6b21247660c98d4b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8d0b2fbe162f672c6bb64964189e9ea
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b9ad4e01b720f930c13d9baee4337935(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e3b3e7eef9e7b5c885e5e24d5a61d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9ad4e01b720f930c13d9baee4337935
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e5c87214b9cd7560a45f2cc0baa4ee76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12, 64, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c719dc4b363858a7ad4af032a5a63e3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c87214b9cd7560a45f2cc0baa4ee76
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 64, 577], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2209112db3199a889a8bbec9c463390e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d02c423db63cdeaf2924d77350aebd66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2209112db3199a889a8bbec9c463390e
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe203a11019c814a0da8c0fa9d13c1f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ece4fa2777e62749f439a9dc8e6529a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe203a11019c814a0da8c0fa9d13c1f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de443b3d29542695cd0700637d147f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e0e8ceda2f0a52cbe4b6f1d46f3bd5e
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20120910b6b97561d8b55f78e796312e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba7a7482e002a5b74974776b13034499
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0cc6368450634f2a92e2d89e5deb5885(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e5bacbfc7403e9e5f68ed28f1d2d0746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cc6368450634f2a92e2d89e5deb5885
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e311c3eab23b97052e69bac072c4e21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42a8bd5dd4855046b829f11f0b40f0e5
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d385260bde5de45225e26f54442cd9f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[872, 218], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_959f08da2637d9d726e8b37924575ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d385260bde5de45225e26f54442cd9f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_17c6ed0c62c6db5d96c4220444630491(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 218], dtype='float32'),
            paddle.static.InputSpec(shape=[218, 872], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16c37efb8d5a765ccccbc6452e5eb34c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17c6ed0c62c6db5d96c4220444630491
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb77ec3dfad1a4af5dd7b22df829bc52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 4, 4, 7, 7, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c51faca5490c3dafc635f1d9c37b3d45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb77ec3dfad1a4af5dd7b22df829bc52
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1042ce13f0e1b14d7fe5de073ac43f48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f67a9e311b4a837bee4925f8de831f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1042ce13f0e1b14d7fe5de073ac43f48
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ad19a8c699ac141b52e86e80cefef3ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e433d22b282736c7837832d36fa992b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad19a8c699ac141b52e86e80cefef3ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e089596245e7f36c78d0c36111796a00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 64, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11b9b369fc2016f0fde58a161cd3d76b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e089596245e7f36c78d0c36111796a00
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_272624b328b598ce8fcf159d97b1d276(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_202ab122a27adf956ef727c819477f8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_272624b328b598ce8fcf159d97b1d276
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f67a9e311b4a837bee4925f8de831f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1042ce13f0e1b14d7fe5de073ac43f48
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60ce36f8d1b6eead74f9bc6deac9303c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_987aa13af058730fe4674c6d0e85eaca
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abfea4616096a0a9c09f84ca4506efd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_089cb118a44d34edb3ac535724f982cf
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f541b52c468302b5fb2f4b3009c5e1fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4215998f47a96645f4f584ca66e52519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f541b52c468302b5fb2f4b3009c5e1fb
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_377b7d49a2a407a2d8037bbb724f4c82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 1536], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38ee6bf2a3d830e1b632cf28ac0cb775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_377b7d49a2a407a2d8037bbb724f4c82
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e8202115e00cd54a9b7b956de0b7f1b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[3136, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25676f83c0271abe11a6985d6f1fd00a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8202115e00cd54a9b7b956de0b7f1b5
    def get_inputs(self):
        return [
            paddle.uniform([390, 3136], dtype='float32', min=0, max=0.5),
            paddle.uniform([3136, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42d4aca5e357d6c53b2b609d9e75639f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e374bd3daac99ba78bc8af729540392
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfe30dbf715e3b6e67d32e96ca467d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_801a1ac71b0650f60862a3ec37c444d8
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad5d132c43189752a6cc186437e09ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30b37d63558c53c0ec0e1dc9de8c9792
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_741453ca674269d135c64335d69cfa53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79edb3e364682a29136fa7785594a054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_741453ca674269d135c64335d69cfa53
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_797abeeb41d8fd95a52da3e52f333213(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2, 32, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c324365a493eb2f985d24597e26cd82c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_797abeeb41d8fd95a52da3e52f333213
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 32, 640], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_69a44f3387d3926c967f3997db9a17d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61a3409031237bc75980b9031fde79c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a44f3387d3926c967f3997db9a17d4
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3434ecb81236d248f5c086e0fa36d4a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d30130da1ddb5967cb3c91b7446873e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3434ecb81236d248f5c086e0fa36d4a9
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5dd30d6e4accc9fbd2022f603eea18e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b2858151b156d1ede2fcff61496bed
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_971f52a9dd6da830058b35926c81fb3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83f54cfd2cbb54040ed22e3633bea1ff
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5401952a09704f25f729369b2df50da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06625f1bd7809d4bd5b7c01f914d78ac
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c011bce0194b8738b4569a3a909ae22c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b2b09ab4307a715cab833c43bcff35
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([86, 3, 64, 198], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33057a60cd818f300bb9a5ca6b72b0a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04f35582cc04e281a68786b43457b3e8
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_22fe6e62d3f42789314fd47e0af2cac3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c4fab6a958c2b9edd772574465764e7
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4420f085df3ce2c65797b89d453314b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_987aa13af058730fe4674c6d0e85eaca
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bc256b8a879005ad7db9d33f98ccede7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa826e92594830e4edf761a256e029e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc256b8a879005ad7db9d33f98ccede7
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8594dc963b89f31fac6f7532d762b849(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81f7d694ef500b75fa6363807da1d829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8594dc963b89f31fac6f7532d762b849
    def get_inputs(self):
        return [
            paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_81f7d694ef500b75fa6363807da1d829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8594dc963b89f31fac6f7532d762b849
    def get_inputs(self):
        return [
            paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7a7742a79733a83469ec013e1b9a0c1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a519bec52bdf4cba18e3ea148900a47f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a7742a79733a83469ec013e1b9a0c1f
    def get_inputs(self):
        return [
            paddle.uniform([11, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db4cfd553e89c1501d4801c16f3e82e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8594dc963b89f31fac6f7532d762b849
    def get_inputs(self):
        return [
            paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db4cfd553e89c1501d4801c16f3e82e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8594dc963b89f31fac6f7532d762b849
    def get_inputs(self):
        return [
            paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ee44adc618b0947b33ae64d13c959c4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 2048], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95520422d16b4b3691c225acc1c415ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee44adc618b0947b33ae64d13c959c4d
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 2048], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_924e546fa7316c3a4f7c04533dd2858c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_900a09956722838f68a0dc2e7d2c28d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_924e546fa7316c3a4f7c04533dd2858c
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bf020894016617b02bf3135c642c7530(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 7, 7, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_defcd1f8f2e752f15c99e2761bd67e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf020894016617b02bf3135c642c7530
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ad6ec1e477c23106aabfee42f49faba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b21e3ae0b0556a504758b0ef5912cef
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cef0d48a9ab11f5636538e8632421b1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10754737f3209fd76fe18c6f5b939c25
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa764ba2bb934690e20cb08af12b1697(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_801a1ac71b0650f60862a3ec37c444d8
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c473c446a1110184aca29cddeb8ea77d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30b37d63558c53c0ec0e1dc9de8c9792
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b11779a4b64bd91628cffd8efee5ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddfa228cfaacd7ffe628a2c2ae74c0ff
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_598028df31cf25171a23d0af0d5a58ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d69680c6b6b46b8dcadfa5b00d0541d
    def get_inputs(self):
        return [
            paddle.uniform([43, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_750b25d8627b70411a5929a35fef192b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3022170efe4ff66b43ddfa0f8282c3b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_750b25d8627b70411a5929a35fef192b
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bb99858ab57898ad71e5733e3ad86ecd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da0b130fab5ddd8320e1f67a35e4cc10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb99858ab57898ad71e5733e3ad86ecd
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8fd85290ba81948a1c0a89e9d693b873(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd2f643d8cf1d2720611ef12e30e39ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fd85290ba81948a1c0a89e9d693b873
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7bddb4d7542f5298360df11ab5d353b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a5b48ff4fddaf0d9d17444fc8faa1a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bddb4d7542f5298360df11ab5d353b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d9332fd328fa05c2bdd9ee7b989df70a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8, 32, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfe0c84567d88374b6f6bcd4a918a031(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9332fd328fa05c2bdd9ee7b989df70a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a4397ac10ac94eee0ae83b5f29534cc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_474f6ede0d38e4fa486ad322492ea81f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4397ac10ac94eee0ae83b5f29534cc4
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd2f643d8cf1d2720611ef12e30e39ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fd85290ba81948a1c0a89e9d693b873
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4b3953f1af77d2de47f3504827ee9079(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1280, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0647f505bf3bf5adb5be533fe52b16a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b3953f1af77d2de47f3504827ee9079
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280], dtype='float32', min=0, max=0.5),
            paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcbd6bcbdd5180bb5e3a0675b4e6adaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42a8bd5dd4855046b829f11f0b40f0e5
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fea2ea94f227862cb57b2d2f26da4fec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97e6c82cbc061a4623e6deb505cf37e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fea2ea94f227862cb57b2d2f26da4fec
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_532cec7e7bdb8e619bc9f558f667080a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea81763322d21cd10e9dccaa15ae2693
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b1b175c4577954f0a45311744110719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afe4e8bf8bba039bdf4c6b915ea202dd
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1826d1389292b374ecb255faa24f83fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94eef12a2b31cdbde24acd18cb974e2e
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b1b0517d917fb2a2def90556585a93f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34c98fdb5c8c3816f9a7d51e9e8943bc
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aaf28c20fa26944dece90d7de0ec0f88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76beb765e1b16d1c417b734b95f6c70d
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e389cf9f3fa5aa15dc6615415362180(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e105e2bd3e16d933cf3aac2608b7b87
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2db31cc34f88aff44a05ad0268338854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9ad4e01b720f930c13d9baee4337935
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13be8c073e94b71bf90d8bf64b0938f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_801a1ac71b0650f60862a3ec37c444d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_301a41c30227a6416c9cc436c03ed7ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30b37d63558c53c0ec0e1dc9de8c9792
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac7b97ef3224ad57e1f5d1f5cd4bc7c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b2858151b156d1ede2fcff61496bed
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_266e2ff6cdbf08c6a683a0f2ac949919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83f54cfd2cbb54040ed22e3633bea1ff
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfe30dbf715e3b6e67d32e96ca467d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_801a1ac71b0650f60862a3ec37c444d8
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad5d132c43189752a6cc186437e09ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30b37d63558c53c0ec0e1dc9de8c9792
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_410c13ed8030780e93ff4d6404e0871e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1536, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80fdcaae4f1619c26fd386dbe87d8878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_410c13ed8030780e93ff4d6404e0871e
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536], dtype='float32', min=0, max=0.5),
            paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f43418c40e490982a56118984237c61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e0e8ceda2f0a52cbe4b6f1d46f3bd5e
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8139ad5b1c8a6c5397494a541214806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba7a7482e002a5b74974776b13034499
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f33a9b17f375bb8ba1a1d28ac94a40e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b2858151b156d1ede2fcff61496bed
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e4beed46ecffb81496634171036692e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83f54cfd2cbb54040ed22e3633bea1ff
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c426a62e273641622bd515483335430e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_410c13ed8030780e93ff4d6404e0871e
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536], dtype='float32', min=0, max=0.5),
            paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c51faca5490c3dafc635f1d9c37b3d45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb77ec3dfad1a4af5dd7b22df829bc52
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f1e493f899b080cd270244cd6bde4ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d69680c6b6b46b8dcadfa5b00d0541d
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c202e8e8363d471005f19305bd2e22e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2da8710becb7326a59c5a12580f1eab2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c202e8e8363d471005f19305bd2e22e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_29646af6af47f7af069cfec3e280b54f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 6, 64, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e5827146d90943e5ad256afc7b5728a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29646af6af47f7af069cfec3e280b54f
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6, 64, 1025], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a244fbda817a9950ffb304b74854c59b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 6, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93a8abba184eac9d7a35682ae5c61ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a244fbda817a9950ffb304b74854c59b
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_28098b72b837337579bb39ca938eda63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82fc1b0e50e9ad48abf0ed115f4508c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28098b72b837337579bb39ca938eda63
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3806e76b27f5943b70995cbabc1d49e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_750b25d8627b70411a5929a35fef192b
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8956df5865d047bffb582ab9c309e0a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb99858ab57898ad71e5733e3ad86ecd
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53ec5730bb1ed0e7d041fcd8ccaeac6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f541b52c468302b5fb2f4b3009c5e1fb
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96be2a2854e54127b32089782f5d774b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_377b7d49a2a407a2d8037bbb724f4c82
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_65a2e1980d6554e79e73cb54a0f304d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 150], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b36eea2682df7b12a6bc0e4950c6106(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a2e1980d6554e79e73cb54a0f304d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_960f0fee850546f6409fa01bf1f438e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c202e8e8363d471005f19305bd2e22e2
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a93c8ba0b50c732c7a4606e6557e63a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e0e8ceda2f0a52cbe4b6f1d46f3bd5e
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_219808d8412d3acf882b5d6c359723bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba7a7482e002a5b74974776b13034499
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5dc5d1afb533e4e100abe238629631d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[672, 168], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2392db77106d0bdf68826b29fb89e388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dc5d1afb533e4e100abe238629631d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_57f842c79fa77199868bb9e7614db7d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168], dtype='float32'),
            paddle.static.InputSpec(shape=[168, 672], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b26a6b29f0345ed65b7e7abdaf0fc86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57f842c79fa77199868bb9e7614db7d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_31c64afb26ff6a5d038e9121249d0162(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0444919b2db07c996c2ea88d99db293a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c64afb26ff6a5d038e9121249d0162
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_be7af5f9ffb5e4897fffa73de2398882(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4551ec087dfbb8b25c29c5c1fd5af115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be7af5f9ffb5e4897fffa73de2398882
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_321e56fee900af0d1c3ff14331ba041c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, 32, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b0c856c82a867db1ff5e423c443f2ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_321e56fee900af0d1c3ff14331ba041c
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_536bd78be4072e580650ddb8cea90eb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb346a530915da85372e7a9b30c25c3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_536bd78be4072e580650ddb8cea90eb1
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0444919b2db07c996c2ea88d99db293a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c64afb26ff6a5d038e9121249d0162
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_739d87ce5582aa2102bb56098d31ca9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fd85290ba81948a1c0a89e9d693b873
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a55dd06a3e19388b7e2595ea76e44d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bddb4d7542f5298360df11ab5d353b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32c0d55df16d354072de909131b56397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9332fd328fa05c2bdd9ee7b989df70a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9118f67cf4b0e228ad21e8b13787fe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4397ac10ac94eee0ae83b5f29534cc4
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_739d87ce5582aa2102bb56098d31ca9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fd85290ba81948a1c0a89e9d693b873
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13be8c073e94b71bf90d8bf64b0938f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_801a1ac71b0650f60862a3ec37c444d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_301a41c30227a6416c9cc436c03ed7ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30b37d63558c53c0ec0e1dc9de8c9792
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f47590b2e9e8c2c9f318c11a03147a6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14a629e06483c2d204831e741fffc750
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_214252e4f9ea802bf22705031cb55e3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f27ab9b93e77c852a668b6fd1d024f6
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae7533286e957ffab6bd2f64a4015a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d69680c6b6b46b8dcadfa5b00d0541d
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f1e493f899b080cd270244cd6bde4ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d69680c6b6b46b8dcadfa5b00d0541d
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5fb7a8d58a1408abc4e48307b3a13c88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47e662f53a5554c6d5f0e9238135b068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fb7a8d58a1408abc4e48307b3a13c88
    def get_inputs(self):
        return [
            paddle.uniform([43, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_defcd1f8f2e752f15c99e2761bd67e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf020894016617b02bf3135c642c7530
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a623050a6253c01a2a0e5ce486b1fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06625f1bd7809d4bd5b7c01f914d78ac
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cbf0d17213d5ab96e0c20911ab16f4dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 3, 64, 197], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f83597a234af333ff74f7a797699ce9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbf0d17213d5ab96e0c20911ab16f4dc
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([54, 3, 64, 197], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_43a2f92dc4f3e6c39b03673d2a936be4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 3, 197, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2561c267cf9af867f23a6524239c22e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43a2f92dc4f3e6c39b03673d2a936be4
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5aab2d9095cf961f7cc3b42d28178efe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_338f1d7b4e062b04c714ed58e98a25ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aab2d9095cf961f7cc3b42d28178efe
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_831b1261bd2922c7d7ef3ede386390bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16a95ad01dd1d4b4d72dfe0af5294b23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_831b1261bd2922c7d7ef3ede386390bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2ef34644bf0ec62f7073cf888b7373f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6316bcf5f88bdec9a8c1c688ef95577d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ef34644bf0ec62f7073cf888b7373f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_88b3a6069682af88b5812927c6487ab8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 32, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d57db97ca561a84e6b3b1a6feb67438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88b3a6069682af88b5812927c6487ab8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_831b108e42826b3fe6b7587673a3c593(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc01cbcff138c494dce786416b1db0a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_831b108e42826b3fe6b7587673a3c593
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16a95ad01dd1d4b4d72dfe0af5294b23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_831b1261bd2922c7d7ef3ede386390bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47de2d1f7a60a44a98137df4106da10f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06625f1bd7809d4bd5b7c01f914d78ac
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_114675e8dec6ea6b21247660c98d4b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8d0b2fbe162f672c6bb64964189e9ea
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4420f085df3ce2c65797b89d453314b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_987aa13af058730fe4674c6d0e85eaca
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa826e92594830e4edf761a256e029e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc256b8a879005ad7db9d33f98ccede7
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2a0b6c5f7c481e8c4858b72527b44195(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 6625], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c12f6f1a361672f426816b18d76e43b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a0b6c5f7c481e8c4858b72527b44195
    def get_inputs(self):
        return [
            paddle.uniform([10, 40, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 6625], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9653d5ba560423ea1b1cf5b8319e1cd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea81763322d21cd10e9dccaa15ae2693
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4cf1fb083fd1679467c5fdcb80423be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afe4e8bf8bba039bdf4c6b915ea202dd
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7631475f03e0210f3f0400a399f07d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94eef12a2b31cdbde24acd18cb974e2e
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c463cc9d8038700faaf953d3782281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34c98fdb5c8c3816f9a7d51e9e8943bc
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c38fc28004e400a5c10ec4609ffba855(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b92940709908d3605519deaf34f6ebed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c38fc28004e400a5c10ec4609ffba855
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6b36cc78ab773df4df05fb8a869abed0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46ca11c2d5a066a17c5b45a506e967e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b36cc78ab773df4df05fb8a869abed0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ac7d1428e15221feb10a172314741173(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 64, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46567862fdf9423c20b7723e4a4179be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac7d1428e15221feb10a172314741173
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_95689ac2d69d53f380c03803a5ab6a76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba960b8f72a53530629f0ec578391b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95689ac2d69d53f380c03803a5ab6a76
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b92940709908d3605519deaf34f6ebed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c38fc28004e400a5c10ec4609ffba855
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53ec5730bb1ed0e7d041fcd8ccaeac6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f541b52c468302b5fb2f4b3009c5e1fb
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96be2a2854e54127b32089782f5d774b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_377b7d49a2a407a2d8037bbb724f4c82
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_89c093bff526d573b30c6674cdc361f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 150], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98abe37e6cacf05a73522c30b93ec721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89c093bff526d573b30c6674cdc361f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ad2ebb3e08a2f7d97de5814f9b38355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_741453ca674269d135c64335d69cfa53
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6658a9c1fc7d810221e30d3bb6fc2b0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_797abeeb41d8fd95a52da3e52f333213
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 32, 200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ede2250e739c377d0df8e67656e1160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a44f3387d3926c967f3997db9a17d4
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60d49bdd8dcb659a5e4fbc2c34a50c8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3434ecb81236d248f5c086e0fa36d4a9
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91b25b4f5faefb4ec02766e9a85465ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9ad4e01b720f930c13d9baee4337935
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bdb3e423ab76c5e9afdf685d81fb9a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cc6368450634f2a92e2d89e5deb5885
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f610194a0e5b13c5658d5a1cf893d5e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42a8bd5dd4855046b829f11f0b40f0e5
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a421fb27353afd32d9cd3546b106f227(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 2, 2, 7, 7, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbf4e93c1be05d160db54ce7953d54f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a421fb27353afd32d9cd3546b106f227
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_932e7381b33867579c86fe17c224de04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45f68271394e827a6a90b85c7b15e370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_932e7381b33867579c86fe17c224de04
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_808edee36f871edc2ddd6a88afd1dd25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad19a8c699ac141b52e86e80cefef3ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_98cb45ef328c15e4390f96c8ba665e54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 64, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b30abcef237a206e0566680bb35cb58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98cb45ef328c15e4390f96c8ba665e54
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_19a633bca5588884edac2b5d46193fa3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df0c38ad77a631394f16f27b8ed9607e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19a633bca5588884edac2b5d46193fa3
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45f68271394e827a6a90b85c7b15e370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_932e7381b33867579c86fe17c224de04
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f9b43f6294bb71790d8010ef9fb26477(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[704, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_487fe86b91c905fef4b2cefb3d47bf70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9b43f6294bb71790d8010ef9fb26477
    def get_inputs(self):
        return [
            paddle.uniform([11, 704], dtype='float32', min=0, max=0.5),
            paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e292b308b2a3da0376c98c101261b016(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fffd5934ad61fcd55c2871119eac121c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e292b308b2a3da0376c98c101261b016
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_28aa2ba3c5a5a36d86d59cbe102870c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 640], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f32f54f26ab8e72b2e712273c74fe76a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28aa2ba3c5a5a36d86d59cbe102870c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_815a662f514c9035864db81ec493f231(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, 64, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b06d79a0ed683b4514f99b39197eda98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_815a662f514c9035864db81ec493f231
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a352a6cfa628a797b949f92cbedb2c7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74a0cf9d62d703263487b90d8f147f3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a352a6cfa628a797b949f92cbedb2c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fffd5934ad61fcd55c2871119eac121c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e292b308b2a3da0376c98c101261b016
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fbaf3a21b46be94f3d69252dd596f9cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1248, 312], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8860d4d83b8893b944913211f435324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbaf3a21b46be94f3d69252dd596f9cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
            paddle.uniform([1248, 312], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_980401b65725faef1b892d119fe497c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 312], dtype='float32'),
            paddle.static.InputSpec(shape=[312, 1248], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b64ee34bd4c99ce5a4cbb0f5dc23cb5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_980401b65725faef1b892d119fe497c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
            paddle.uniform([312, 1248], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ef21d714f849b97d294cae83bcd34ced(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d8471edd5c9951a0513c721ab2f6449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef21d714f849b97d294cae83bcd34ced
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f97478aa1d3b49bbeda8ea7dab7ef4c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76beb765e1b16d1c417b734b95f6c70d
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4eb7646c6cc7f429e4709f1fa1c3ce4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e105e2bd3e16d933cf3aac2608b7b87
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c3acadf6da6f9b620d70396cd05a08a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_750b25d8627b70411a5929a35fef192b
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3a6f0a3d55c930730ec41eb54a2c18b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb99858ab57898ad71e5733e3ad86ecd
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_74fc7a88a431358d204459f3828b3992(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 7, 7, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_297bc3f361c14bb55e6ec57df87fda5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74fc7a88a431358d204459f3828b3992
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bc9c68e94c153276787f75d3c3af386f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 2, 2, 7, 7, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f062d6c4fea80dce880fb21ed2b4f1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc9c68e94c153276787f75d3c3af386f
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_297bc3f361c14bb55e6ec57df87fda5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74fc7a88a431358d204459f3828b3992
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c19da4f791842d1ab7d331fefab39ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9ad4e01b720f930c13d9baee4337935
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc2c96179304f83fdcbab12bccce0ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c87214b9cd7560a45f2cc0baa4ee76
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 64, 1025], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c20387fd20d64da827ca5bee5b7543d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2209112db3199a889a8bbec9c463390e
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f5dc62c89c36c584b2bf0c6a11755fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe203a11019c814a0da8c0fa9d13c1f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f062d6c4fea80dce880fb21ed2b4f1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc9c68e94c153276787f75d3c3af386f
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5d4968971be5eed5ed715d3f605267ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[156, 39], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b027039120ebf68ed5a6835026d234d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d4968971be5eed5ed715d3f605267ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            paddle.uniform([156, 39], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2b42f4824eef1b310515c5b296277869(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 39], dtype='float32'),
            paddle.static.InputSpec(shape=[39, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5db250df71091fbf24a1533b8ae8bdc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b42f4824eef1b310515c5b296277869
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
            paddle.uniform([39, 156], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8b5fd7789483438d1311e20b5a93d1cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3c08da84d26e8c41fb297eb80a61be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b5fd7789483438d1311e20b5a93d1cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8381f2585afb8604d7b92bb4d3a9c06b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d50f306747ff1b6884d20149dea6d296(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8381f2585afb8604d7b92bb4d3a9c06b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_19719e99fece9396d7764d9c7993376d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8, 64, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4212419733b5d249644d630c1c5175b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19719e99fece9396d7764d9c7993376d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cff693a67c8decd5b0c72f405959e3e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c361eb35ab65591fd133985042a2b79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cff693a67c8decd5b0c72f405959e3e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3c08da84d26e8c41fb297eb80a61be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b5fd7789483438d1311e20b5a93d1cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_959f08da2637d9d726e8b37924575ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d385260bde5de45225e26f54442cd9f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16c37efb8d5a765ccccbc6452e5eb34c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17c6ed0c62c6db5d96c4220444630491
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca9e81d781f681a2e686b07c549e6385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ce5439a2ce2d3081c3549ab197b62a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf4e93c1be05d160db54ce7953d54f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a421fb27353afd32d9cd3546b106f227
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2eb5e667702c8e1ed2e460608a51e9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef21d714f849b97d294cae83bcd34ced
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_433fe23ad7461e37c82aba45ea741a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76beb765e1b16d1c417b734b95f6c70d
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dfd2f668d260e0a0c3ce16d3be72616d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e105e2bd3e16d933cf3aac2608b7b87
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b3a57687baa34d0034f36394cf4285c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76beb765e1b16d1c417b734b95f6c70d
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_22ed7d521c019710aba3ad7d04a65fb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e105e2bd3e16d933cf3aac2608b7b87
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4215998f47a96645f4f584ca66e52519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f541b52c468302b5fb2f4b3009c5e1fb
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38ee6bf2a3d830e1b632cf28ac0cb775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_377b7d49a2a407a2d8037bbb724f4c82
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b430cb11a61be6e7c5d336367b448294(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 37], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa228e0379f04072c74ea6e50bd03cf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b430cb11a61be6e7c5d336367b448294
    def get_inputs(self):
        return [
            paddle.uniform([10, 25, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 37], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6d1bb9691c26f21db44282c78bcd216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_750b25d8627b70411a5929a35fef192b
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e50d44240a4c6560b1abb20111e9909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb99858ab57898ad71e5733e3ad86ecd
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_74cadfb02762773636caa080cf6ae577(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[120, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f10a7389c42caef740904c41cd36662b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74cadfb02762773636caa080cf6ae577
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 30], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_43decaf0af47e9ebb1c5cbedae9945f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[30, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c02d61ab4ec0227cac1ab6d945e073c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43decaf0af47e9ebb1c5cbedae9945f9
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.506402492523193, 7.807661056518555, 8.065888404846191, 7.2947492599487305, 7.78436803817749, 8.1944580078125, 7.6481032371521, 8.243966102600098, 7.196224212646484, 7.1795573234558105, 8.857290267944336, 8.459726333618164, 7.799643516540527, 7.862877368927002, 7.898179531097412, 8.240922927856445, 7.324423789978027, 7.7251386642456055, 8.335378646850586, 7.039722442626953, 7.731579303741455, 9.243419647216797, 6.439398288726807, 8.070255279541016, 7.706521987915039, 8.942200660705566, 8.597922325134277, 7.838735103607178, 7.5404510498046875, 8.566326141357422]], dtype='float32').reshape([1, 30]),
            paddle.uniform([30, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_18130251c6ab8e3bd97d9fcd0282482d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_397b78630605a996bbf88aa93e97977d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18130251c6ab8e3bd97d9fcd0282482d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b206b11cbbc0d573775ad03248c04140(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28aa2ba3c5a5a36d86d59cbe102870c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bf89e1bd099a03ad1ade21c145b59ee8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, 64, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_086747969ef491f39449d72bb207fc8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf89e1bd099a03ad1ade21c145b59ee8
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_54de99afe1e155f7379908d27db9c802(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9320b43f55b8d044644e5aef8f61c13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54de99afe1e155f7379908d27db9c802
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_397b78630605a996bbf88aa93e97977d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18130251c6ab8e3bd97d9fcd0282482d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3d17b1a5cb0dafd09b99fc43a6809475(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4cbb82d60c9aeb16e48dd94db57a2b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d17b1a5cb0dafd09b99fc43a6809475
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_61808824a3a86add04434ceb6a4c2296(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be7af5f9ffb5e4897fffa73de2398882
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a03b3aaa8b89bd9d28ef785647bfbf46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, 32, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afdc4687fbe0f59d044c002601da3d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a03b3aaa8b89bd9d28ef785647bfbf46
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_510ac85a636f8ea0866c9703ea209abf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f9705aa8b898e89e406ab49e523f5bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_510ac85a636f8ea0866c9703ea209abf
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4cbb82d60c9aeb16e48dd94db57a2b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d17b1a5cb0dafd09b99fc43a6809475
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2392db77106d0bdf68826b29fb89e388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dc5d1afb533e4e100abe238629631d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b26a6b29f0345ed65b7e7abdaf0fc86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57f842c79fa77199868bb9e7614db7d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_37bf22e58c07c5e4eaa5c732c8ebb619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42a8bd5dd4855046b829f11f0b40f0e5
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe1fc53b790109336f05e7554fdc708b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0d33df3f68ae41bbd94de1e59b80100
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fcffb7d673e8de1515935253b939867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b5fd7789483438d1311e20b5a93d1cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4354c1724a50c32c089dd729579770b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8381f2585afb8604d7b92bb4d3a9c06b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e627da7d6cd4d092c08d768572438e55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19719e99fece9396d7764d9c7993376d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d60dcc31a928b793f0e568fcccbafc43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cff693a67c8decd5b0c72f405959e3e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fcffb7d673e8de1515935253b939867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b5fd7789483438d1311e20b5a93d1cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_772ad51701ec01154322f7661efd9779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06625f1bd7809d4bd5b7c01f914d78ac
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62fbf35cb3006f7238b33db636f66461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f541b52c468302b5fb2f4b3009c5e1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c00f1dd70597e2c1e7ca20b05e107719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14a629e06483c2d204831e741fffc750
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e45d7d8133f242a266f1803e370b05c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96288a25145efa7447b500a4ae2ae028
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcbd6bcbdd5180bb5e3a0675b4e6adaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42a8bd5dd4855046b829f11f0b40f0e5
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97e6c82cbc061a4623e6deb505cf37e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fea2ea94f227862cb57b2d2f26da4fec
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9869199676e0ac85824a8b26a76f6590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c202e8e8363d471005f19305bd2e22e2
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_13326ad7b1ac3e391229908de738cb6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d9129a9f98d27b42ed2f3203c1c91fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13326ad7b1ac3e391229908de738cb6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2521e864ee4c411c4471e1374ce65191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b36cc78ab773df4df05fb8a869abed0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b044d69f10ab7d00060bd03ff9ef70c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 32, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20ca766980ab0d25ab6352331b21c755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b044d69f10ab7d00060bd03ff9ef70c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cef1497cdde38d2d6336fbc59c16f2e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c77d897eb009480c2e89f240b023f33b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cef1497cdde38d2d6336fbc59c16f2e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d9129a9f98d27b42ed2f3203c1c91fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13326ad7b1ac3e391229908de738cb6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c0a7d3a1554690d7df34c6353ee0bc1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed943f56282a5fe5bd6c94734ceb3e60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c0a7d3a1554690d7df34c6353ee0bc1
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46ca11c2d5a066a17c5b45a506e967e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b36cc78ab773df4df05fb8a869abed0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ffd761d256119b38829e07b07a7c30c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 32, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ede886e675a51c5b2e11d487b6f1eb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffd761d256119b38829e07b07a7c30c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd6ca0ee8d78181300435495f21af0c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aff70854cc9a0755ccecb0e67c2219b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd6ca0ee8d78181300435495f21af0c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed943f56282a5fe5bd6c94734ceb3e60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c0a7d3a1554690d7df34c6353ee0bc1
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce9e93e6394def3950c1b199264b05a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06625f1bd7809d4bd5b7c01f914d78ac
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8c9527670835c09c55552a9e2fc8c1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbf0d17213d5ab96e0c20911ab16f4dc
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([86, 3, 64, 197], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_afd8cf01e4af3be543ee67020ea76d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43a2f92dc4f3e6c39b03673d2a936be4
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98c03e2097d2a716caeab5ba64350818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aab2d9095cf961f7cc3b42d28178efe
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8e3f65ea6bfb1e32cba42bfd881030de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ff4c26dcd9b15e80dea7601f57d91fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e3f65ea6bfb1e32cba42bfd881030de
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8dd25aabfddd3394c0f4345dc6e8366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ef34644bf0ec62f7073cf888b7373f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_547a231192d287a56d442f8e9d44ceee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 32, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15a96fb54a9a196cac0ca206460a27dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_547a231192d287a56d442f8e9d44ceee
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2ad0d54183be2a43d2d076af2f1ae839(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1dc89ca94b52f841c1aebadc28a5e2f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ad0d54183be2a43d2d076af2f1ae839
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ff4c26dcd9b15e80dea7601f57d91fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e3f65ea6bfb1e32cba42bfd881030de
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9a633fde9f800dfa517a76b714a0c631(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81ba94d707dccb0680c81d03b5c9b3cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a633fde9f800dfa517a76b714a0c631
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d030dd96302c6928b8d5ad4c9ac9e1ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9332fd328fa05c2bdd9ee7b989df70a
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 32, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8aaec16a24a29f84c3222cc8463d6b3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4397ac10ac94eee0ae83b5f29534cc4
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9e22dc4cf592e078431eaddea841319(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fd85290ba81948a1c0a89e9d693b873
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c42633ad6997d240567802961ebb9dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c202e8e8363d471005f19305bd2e22e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3c1bbc38976f9a1809b0f9617f83b28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29646af6af47f7af069cfec3e280b54f
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6, 64, 1174], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62552d84a06b5663d8f7d8facc7fd59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a244fbda817a9950ffb304b74854c59b
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ed641ebfa394e77eb177948e174f8ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28098b72b837337579bb39ca938eda63
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1038943aef7be1605d067e039f11adbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b3953f1af77d2de47f3504827ee9079
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280], dtype='float32', min=0, max=0.5),
            paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3d935d752c5789d760a06efe097760b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9b43f6294bb71790d8010ef9fb26477
    def get_inputs(self):
        return [
            paddle.uniform([43, 704], dtype='float32', min=0, max=0.5),
            paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82fcbb7e46aea3fd1c19245cf58a40ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9ad4e01b720f930c13d9baee4337935
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f70887580ebea7e913f4ec892c4078d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c87214b9cd7560a45f2cc0baa4ee76
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 64, 1174], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69d3a5d78388c5700942705e1abde4a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2209112db3199a889a8bbec9c463390e
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b77dad57f3007ac8b291ec67cdc2cc08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe203a11019c814a0da8c0fa9d13c1f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7e87fa1442ed1627dbe691c5b398efc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3f2ec3fa9201b036ac9b1308620b756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e87fa1442ed1627dbe691c5b398efc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2521e864ee4c411c4471e1374ce65191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b36cc78ab773df4df05fb8a869abed0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ccf887323dc62730808bd283f552e324(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 64, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99a1a1b72f980d89955a87786ae6c12d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccf887323dc62730808bd283f552e324
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_21a6facd518ae97d264a0fb8cac329f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a33e8324253c2cb197f44b0d6c0bcc9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21a6facd518ae97d264a0fb8cac329f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3f2ec3fa9201b036ac9b1308620b756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e87fa1442ed1627dbe691c5b398efc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bb14295388113353a3b226a29a9cab59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[624, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abcd3d4cd5cf46d92f135cc1e1628a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb14295388113353a3b226a29a9cab59
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
            paddle.uniform([624, 156], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_496f3884b33cf56c90f1bdfb4a198a74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
            paddle.static.InputSpec(shape=[156, 624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_662985332a106fac7db0c1cdc72c8acf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_496f3884b33cf56c90f1bdfb4a198a74
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            paddle.uniform([156, 624], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4da1eb82e50e181d8bcbb96eaeb86987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a633fde9f800dfa517a76b714a0c631
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba4fa7b755e60a9509afc728304b9cfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9332fd328fa05c2bdd9ee7b989df70a
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 32, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cdde9fdd7bf76ec969f8b0627971e47d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4397ac10ac94eee0ae83b5f29534cc4
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd22a31be3cdd40412576c2fd347ae76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fd85290ba81948a1c0a89e9d693b873
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()