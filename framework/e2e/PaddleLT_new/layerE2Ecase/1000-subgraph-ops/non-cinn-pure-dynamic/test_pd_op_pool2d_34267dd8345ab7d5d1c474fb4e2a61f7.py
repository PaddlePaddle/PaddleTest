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



class PrimitiveOp_fb27a18c852ef371fa896a7617186027(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1]
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02e81eb09c6c8c6046ce148ad5ee8e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e34f836954c5c726814282a6cb28bf58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9259509f3cac7452ef4c5c5fea587b9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_18772a58ade29876fa8c56dd6c6d656f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cee6821f2de498de0e53d903a3dabdab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_76bd5f983f93b07c88ce5e61e33db8ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_33a3e9753345e4a7920cc05c9ed5ffa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3323c11529a73450b4269c2d790fa2c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f9bfb6d1d33a8809291ed1c5e1200d5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e2accf0e2019fabcb05e36f6422ac79c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6118e6ca335a3c982e2e98f26be07bdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [5, 5]
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c933c86d86174084fb208c2d7b0b693d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [9, 9]
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8df0d36a111f33a9e7f05fe0d6b253ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [13, 13]
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c097c5dacde7f4a57a1a70dea59340a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ff025a01627778117be92cdd4227f752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6a1fc1d8f40c80923ed14416f98faecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e20fb5babeb1e25ffd0d35fab63c0c83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_92ccb64081babb5bf91c3602106775a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7b79f3d4bd9c82714414dda7d1aed1df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9525eebc70fadcea61883f187c891ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0c2b6bb633950ee8f5d5f5ba17b0b78c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f52b5fa871181781b43f0d05600a9ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ecb7c0e7bee0c6e673559759497b0364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3edb8f57148bbcc4fe2fe68d2b6390d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9fa49bbc8d47796f31a4a74321eba26a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1551dc93a7b06f99265af625504738f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0b681a5b7e772df96854d9bc37175261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6585140a180d2197cb618d452a9a34ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c6f279ae52640983e8c79bd89524dabc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1]
        return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18bb1ffe0558166e63b81ecf104d34f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1366758ae0c5843daa7eb7b62434123f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d4447afd92a1c66536bf0a5b290e92ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_685686a3c0ea51f1157d3e53936c623d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 2048, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_77380ccf45859d2fcd026fcfb103be3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_427ef091af7ca2e1c2aa1e3a1bf285c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_473cadc00939a9270a0b083091d42e23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d50ed8d0c6970b1a2a569adabc8d2ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e37c5e3b1912df34b2c1412c70acf0ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c92bea6609fd9d3180644f624639ead7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([10, 1024, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ddfd2def4eb09824c1bb0dbab51d82aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c630eecda563108f8ad6371166377c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_722cddd8867ee9123e20deb900773bb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6306a4fb99fcf2c3e693c2efe1f3ae90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_226b0954bbe9a8c829a2d75164e57d59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a1151b2ebb4266d8852df2eb5ecd22ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b05a8f4d77909eb35314f8cfcd57e711(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 2]
        return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b22f6eb3863656398a4210ba38b66b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b05a8f4d77909eb35314f8cfcd57e711
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_fb9e0c241565c463e8f1be09b7dc6e5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [4, 4]
        return paddle._C_ops.pool2d(input_0, input_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a46106ad895fb1a704d5755040e74f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9e0c241565c463e8f1be09b7dc6e5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_780fff5fef0b93be2bef8bc99eea8b9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [8, 8]
        return paddle._C_ops.pool2d(input_0, input_1, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_224e79f6d6749d194db3ce8c50481bb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_780fff5fef0b93be2bef8bc99eea8b9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_941ac75906eb775012b787a3a43f625b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [16, 16]
        return paddle._C_ops.pool2d(input_0, input_1, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ddae6a77d2679dc8204c9d8765b4cdfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_941ac75906eb775012b787a3a43f625b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bf6df1a0217d23806f2572782cda3ae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4444b00e4485b5ffa98742877bcc7df6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_40cbf354c2bc1d4d63f2e30b15863fec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1335794b6857b3dc7ace167b2c68745a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3498b036c2223248a857990cc0a0778a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1690154a3f1fd3686ff077c55d822f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_941ac75906eb775012b787a3a43f625b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f9bccbb2c22f908e7c7a48a9702e0087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_780fff5fef0b93be2bef8bc99eea8b9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_aabf71293af741636c14040f048690b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9e0c241565c463e8f1be09b7dc6e5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_af50320eb72c641697b5f32f33a6fb0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b05a8f4d77909eb35314f8cfcd57e711
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_60411cfeb537b2031a4982afd7254833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_941ac75906eb775012b787a3a43f625b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a18c824bce7065a27d7b9d29e8e8fceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_780fff5fef0b93be2bef8bc99eea8b9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8126640a9350c63eb8ec244cd44ca9e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9e0c241565c463e8f1be09b7dc6e5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a31626e3cc82793b570aa9fa4d3e60f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b05a8f4d77909eb35314f8cfcd57e711
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e6b9e441b291bf9f2440825e92e336db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c2f4094cd3c3f5b364b164654664228d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_33a3e9753345e4a7920cc05c9ed5ffa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_830c213ee59c7f05cbfbb5e1d7cd04ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e81f56ddfef9a13f2f6f2da520202a29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d12c2541fd99149df4a44e5e5331ccbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7298abecc10e4cf12c30b5fb787c5467(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_be39ec04cc71a7cf14d1c15d93c254aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c292d82af69bf336c732472123d31d20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_272e9d7850a6f3ce86a93fa7a9466df5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b5479563c2f5be5cff111c74871a2d78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2bef78bf0661e6beaed4515135d752f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_78ea26cecf5b6074658581620a4a7532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_242cefd937a70ba8d052d72cdac3ad29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7d0ba94f816fe42150562db14c292b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a9a0e698b9d411fefb1ed1c2e7895a49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_31bb3579fb76d3e6d5207d648de34418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0da0170c90ae25820d1102940d29ce9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_12b90d629159aebcfe05c2fa8ecbb7e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8d44bdb2b93d0f943928a3250845c7fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e5fe6bddeb5c7cb6b2c69c5c2f5fa6a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_29ddc0c3de47872493aaaae212f446df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_08313a7a5ba55b39575f23f462c31760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_81f23c52c7ea2b2a414b22ef5b4deada(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a12457278b20cdc7da77a2cfa5f452ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d063bc08d9489812fb20882aad11eef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b649611e00f59d7a76753bdb2d13def2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_826fd104080154389e401821dbdd7a0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_08ae74b19e6c084f9645ef109b3dc110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_09ec63f93701c6d928b072dc15eadbac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e6ef270a4d07351e54f1cbead0994e10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0263e3436993a3d5df03136131b56161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1200fbe1ebf353524693e41f2602cd22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 320, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6a13d300050f5a93ed6ad7b30f22e0cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b30e1dc391bc3960cdca0f7ba902ec45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3, 3]
        return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f00512f489a1b661cbab6f9c024a8493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30e1dc391bc3960cdca0f7ba902ec45
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_10358ddd4bbb170b4fc2ba94fc6bd9e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30e1dc391bc3960cdca0f7ba902ec45
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_308484cde5318b0ac6cb25e0ea5d1e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30e1dc391bc3960cdca0f7ba902ec45
    def get_inputs(self):
        return [
            paddle.uniform([43, 512, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b95be806e5caa4c46ccdce9abfcee99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a73a19a0342cc51c8bce351b4e2fb447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5fd87ea9dcb6675f680583cc3e0a3aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b71e8218f982f54d9c3d5f442e70fd01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0114859f285db1abbfe0556a3737d3a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_edd66266ea83aa7b4193c90041441580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_41ac2e796e5489018911ac4e47ecb56a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d1d7ab6f9b8c0973d219ea548d36addd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f1f44aa1736e3b0ee83818dd4bc2435a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 2048, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d44e25fec490cb42d6706c3b06dd9207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0e7de785eb348e96277919f20ff1da0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_842e42001f4ee210ee1f6c6e5177b205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9fa49bbc8d47796f31a4a74321eba26a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2335e31f3c49813baf73ce25e23b8a5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1b58ef750922d04336cbbf997188114f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30e1dc391bc3960cdca0f7ba902ec45
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_63da794c8203726cf15102b2dd4639c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30e1dc391bc3960cdca0f7ba902ec45
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0665decfd83a190ec15a43bb108876c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30e1dc391bc3960cdca0f7ba902ec45
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_18526efa7b895f90bfe49dd2af2c41ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a8359da8f21a821b83f0b547278fd0cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9df6689a6a8850b34c829f60d22ee78c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_450388df48ca255416a3c693ca3f4a5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_780adc3da919a04fce3d5c17b85be1e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_10314b2d6dc852daf063dd1b4d3970e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [7, 7]
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1a3a3123d5e38a6ecf71cd8c9056e60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10314b2d6dc852daf063dd1b4d3970e6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f1a3a3123d5e38a6ecf71cd8c9056e60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10314b2d6dc852daf063dd1b4d3970e6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ab4d71d7dcea57d1345b3855ee831ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10314b2d6dc852daf063dd1b4d3970e6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ab4d71d7dcea57d1345b3855ee831ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10314b2d6dc852daf063dd1b4d3970e6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ec99d2ad6dd1329d06db2b3bfc64ce59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10314b2d6dc852daf063dd1b4d3970e6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ec99d2ad6dd1329d06db2b3bfc64ce59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10314b2d6dc852daf063dd1b4d3970e6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ed8aea8b86bbde74b343c8dc958a5bf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10314b2d6dc852daf063dd1b4d3970e6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ed8aea8b86bbde74b343c8dc958a5bf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10314b2d6dc852daf063dd1b4d3970e6
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_48af48ad4a3ecda7abe3cea9c317fc24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_6d2f55725594ec7af64a761524811689(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [28, 28]
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f49a1c3ef754464e32a8e99075913876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f49a1c3ef754464e32a8e99075913876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c1b799f0abe7da3db5f8b6759308d91b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c1b799f0abe7da3db5f8b6759308d91b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ff50db1382c02433131dee723d22457d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ff50db1382c02433131dee723d22457d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_54eccb159b7f707e9798ba0f6c84c957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_54eccb159b7f707e9798ba0f6c84c957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_abf1417f05f54a15c9f9a1aaa6bb5922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e6b228d3f45b509926b5ae409bdd1b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_08ae74b19e6c084f9645ef109b3dc110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_aab00d848aa751f74488ee2508f15227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_118e7c4d199f356313ebeb7ec34eb73f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1102a57102e8114bbe302b13200845a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f6629c9e47fc634ab1f9ebe3b6a0cb77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0c76d6fa5988c61b04e644b293d55d10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8a2ee94292351a17ff3e2ec867ec0166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b7736468fea1114a4b88e21ff2a12d92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [14, 14]
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53f8b88511493e440d3fd6dcdfc6bdc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7736468fea1114a4b88e21ff2a12d92
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_53f8b88511493e440d3fd6dcdfc6bdc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7736468fea1114a4b88e21ff2a12d92
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f21905953e03f6663f8fa4116d1b7cb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7736468fea1114a4b88e21ff2a12d92
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f21905953e03f6663f8fa4116d1b7cb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7736468fea1114a4b88e21ff2a12d92
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_69c85cf269d5f6e102346f88a138e746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7736468fea1114a4b88e21ff2a12d92
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_69c85cf269d5f6e102346f88a138e746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7736468fea1114a4b88e21ff2a12d92
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1e49d93fdeb7751314385fff0cc10fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7736468fea1114a4b88e21ff2a12d92
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1e49d93fdeb7751314385fff0cc10fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7736468fea1114a4b88e21ff2a12d92
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_fe7f4de2d4357972876522e925b9c93a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 2]
        return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_159d17d624368980479be8e57b4e8caa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe7f4de2d4357972876522e925b9c93a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_be66bbde6708261aca9cd0477a69d0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe7f4de2d4357972876522e925b9c93a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e89c8c19413d0a55ab833730f23f7ac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe7f4de2d4357972876522e925b9c93a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_061604673dffefc26f71c2d843efd825(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe7f4de2d4357972876522e925b9c93a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1d600e917e8cf8c3c05a32987273bdd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3, 3]
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [1, 1], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee7a8ca2071dc157aa86ee2b5e8f3668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d600e917e8cf8c3c05a32987273bdd5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0d7b400c2892a66a22055a1ce9ef2f15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cb6bbfdafc1446470840f073e19ebd54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_aa7319972dc2f13d4d2df8e035401f09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0997aee1c3b64aec2f0f5ebd6529e311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b05a8f4d77909eb35314f8cfcd57e711
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7cbe59dd804c58610d83c3dcdadb4c59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9e0c241565c463e8f1be09b7dc6e5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a47eb028a81b46ac946fa2ea5b9f89f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_780fff5fef0b93be2bef8bc99eea8b9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_aec1253cf680e26527374cac285d7548(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_941ac75906eb775012b787a3a43f625b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2bf853ff71698fe1904b371bddfd2aa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_65dada94b97e6a20de04c9161a922bc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_417988e05f71e10b2eb9373d557b8fe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_82580a21a2fa4880e96ebe804dc37a44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2dc888af68678f804297962e7d0f4cce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_5e3ac6186b26d34c6807a8f6a73f556d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 1]
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NHWC', 'avg', False, True, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9853666c6a8fa4c697ad215833f1c60c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e3ac6186b26d34c6807a8f6a73f556d
    def get_inputs(self):
        return [
            paddle.uniform([22, 7, 7, 2048], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_40b86988d886533835e6132d1e6b029a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_75fb805ac74dc4e062752d370dc167b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_631f52d4026601935be959015ad137bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_30485a276c6a68f84be77671b7e6085b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bc30af3c31ee4758c955c90d52369d45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e4b176747819db0471a737e5feb643f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_de0ad7590ab83067a836a4cfdf9ea72c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e611bb6afc87bb69506dc7e3170e4a3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_48af48ad4a3ecda7abe3cea9c317fc24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_45294aa95185a418c4ee0d1ba6e65c11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7a93421c08ca23ca4d94bfa95d25e9f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cce635088fd17f1580f55ee7994867c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a1478fee938c60878bd2ff278ff4cef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2e1326b2aeb7ebfe3d0195c5a9f3bcb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_69f56edb79cd69bb911a763e7e7dff36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c2544b70876811d89ad930f1e7e37ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_736c7fe833b02ca99d7e8ee421918fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_363b1061f195a639ade2c58557e7d412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30e1dc391bc3960cdca0f7ba902ec45
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_beaf0ff0693d6d0343a2e447ea139e25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30e1dc391bc3960cdca0f7ba902ec45
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_811f826368ddd7fa58d1269685eb2aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30e1dc391bc3960cdca0f7ba902ec45
    def get_inputs(self):
        return [
            paddle.uniform([11, 512, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c65b77f589019e89dc3c586b36d925c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b0e752a3cb0d98b083a2ec2c7f5c9295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_32fc60cbc6c990d72286f75fcdd41308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7354502781e875684b627c9dea5299e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dd3e72c4f2b5be4e8447721f7e523c60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d06f3193b4bd2adee9981bd0ad365e1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bf98eb4cefe19a022749cf93885357d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5fff4ad2c16bbc75d0349cef110ccfe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ec8ae00f7eab14ed425801e0cf154df2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_792781a69ededd5918802cc39e288c56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b02de876e266ef6ec7474ea34e675b66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_967a61058d3837424daf626029621910(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c22e57c0cf0b593ae20adcdd5f33163c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6c73676143240f3c451ee4339b9f8c72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fc9bde7b4a492515382e0d8807bd70c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_551366a423644d0a57e5d0578b324767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b05a8f4d77909eb35314f8cfcd57e711
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b27d4e1091888e611af95cc313fb8522(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [9, 9]
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a074ddd791d8f3e4481cfb9e2d1a9fd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b27d4e1091888e611af95cc313fb8522
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_99b4a4844c938c91594198b9378f47ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3bf60042adcb6990d91f9a7ade77f2b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5a2830e991fc35abb1311bd1a3e09756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b8cb0a2d2e6a665c510d23dd5c47e98a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_00ee3c0d3906fbdd58d95660349bb102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_552f52ba0c9d250f9aa4533430eb3ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e2ad99131472dd135d7c815ffcdb3e9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 320, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8d2903ce0c8ef729eb33d26cccd04727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eba25f82f1398d9f8d8c709a4abf29d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d453ad6f585b4e541e991912bb43a112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_35f715fda82266b85f28ca91273607cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_17ac288bed7ce7208b563c25a44fb891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f74eb2100ec202c9372f4b5e1aa6e64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_baf87203f4eeed2c7160845a8bc4d048(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_504d5bdbcf7136b50124db595e55633f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([22, 1024, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d5a9e67d58b68f4ef6ec87ba6219e576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2de2d45311692af5ab9903d94618b2b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5fff4ad2c16bbc75d0349cef110ccfe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_441faf37755c05f6dd14ba70777a2d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b27d4e1091888e611af95cc313fb8522
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7ddd81a2c7d6d80c78f90537dec77940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bff05afe299e4d7a000d7c023396cfb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_01c9ab61652f762c8a3a1384ad455a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d1d7ab6f9b8c0973d219ea548d36addd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7a6fd78a0f901b23b585086b84d71ca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4f9705057ac00068629e595d00970bbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3393bdea6d4839cbde7e2fb2354fad03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9925f4e3bf24014ef3942d24442afcc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e564cac6fffbb1000b62b4f9a9b5fe9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ed983be4b82f3773a85bf1c900405c30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ca56a8079ceca523cfe0aeefe48d386d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0852aa94cb0d2a05d7395e1bfc963d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_118e7c4d199f356313ebeb7ec34eb73f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_87e8c76d0c7bcca992ce7dc2039a3f4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a1478fee938c60878bd2ff278ff4cef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0ec493069d6075ecde33b9cae12dd0a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d1c2030bc7c81ed02e6a86902e121d99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4c1c35861c1b3100e351b861efdaa19b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_627a5f9d90a0acb1f1aa1f3dd3f36942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7e9dce6d8a654045799d2c80134d56aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b0908d7afe97449c0b47856282681bf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2230dc3ad2c06fbc8c02e29f41233e5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d3ff8f1503f44a00b464efc1fd9cb8ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d73674230787df7673d883d70c3c0af0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_038341abbeb6f3e5a92f0b09c6518340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_596d6e2f6e240d86b89fdab97273c05f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f52b5fa871181781b43f0d05600a9ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_4aef48db78deece9c1f3fa8b7cd7c2fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [7, 7]
        return paddle._C_ops.pool2d(input_0, input_1, [7, 7], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa9ece4c54af624e5b54e16a6d9df4c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4aef48db78deece9c1f3fa8b7cd7c2fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 704, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9525eebc70fadcea61883f187c891ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_302b6312ab28a5b9bd506954652d436b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5796ec24b3a40385e660c7a96e3b4eaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_90fb74d431c12aa6050c24b7612131a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e2150acc46abbe357027c0094a5834ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1248bcf525a75b53d1432b781c9fd75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bbb533f86a91b6bf4b98e000c72b6a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bbb533f86a91b6bf4b98e000c72b6a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_08250f3a9f9ffdd1f92ee1d7855fb3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_08250f3a9f9ffdd1f92ee1d7855fb3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e7bc4ab7590e45e3c3b2715d3eff0313(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e7bc4ab7590e45e3c3b2715d3eff0313(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_10c19d424708d257493d3656cc7721ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_10c19d424708d257493d3656cc7721ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d2f55725594ec7af64a761524811689
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_920098fc5b70d34973a669d8a01974c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fa41881b1aebf95736f15d1de75d27be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_13f956d202cd472e72d12b095354188b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_61a390fb5847b82519395109dc98478b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1c43df10217967b48e71e17bbfa223f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_08ed9d59d9a3630134196563f1302c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f0cc849edbc86c1971fab67feae8a138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ef0f8a6a7a2b91428a211d8fb44a0e1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_59b617b7f01011e48c3f8679eff8021a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f0d8fa1c2a3cd45f68aca1e783a6cd91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_941ac75906eb775012b787a3a43f625b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dae30c84885d9cc5d9aa26879df71e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_780fff5fef0b93be2bef8bc99eea8b9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e90ad2b0152408158a3b5cf4f0bf74f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9e0c241565c463e8f1be09b7dc6e5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d7261cafe2b7500153113b06e3dfcc25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b05a8f4d77909eb35314f8cfcd57e711
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6a1fc1d8f40c80923ed14416f98faecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e20fb5babeb1e25ffd0d35fab63c0c83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8dd45eaf58a2e05d2e69522ee1a43f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a93aa9fff7f59b7fdd428a211d0132b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6b038aaad3d1566d8286e1b8f46499b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9a6b10a166ba1b366952e3d5ccdc2143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c73259d6a12f54a325161c031f94a030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f1b4b94769c0b626bba60b305ad2a385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d21492bce7df6e56aa247eca40761872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4188374a413ad91406c31c0217e9acd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_83e0580a5cb51ade94db211f0c1e3862(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8dba384326c0dcbc8e3bfe49d1a7d54c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1602155e9b204162ad8c95b824c0b527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5fff4ad2c16bbc75d0349cef110ccfe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_71a6b39dd6f598856a94a581ef9d3aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c166757e54936fe98e2ba0d564bbb8bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d757e30d08c75a1edc91591ee5a47dcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_20266abdd04e4c26fb09107084864422(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f72337117471d140caee4c488256f97f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_febb88470cbd996dafee90695b69118f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8e931d4569b8db0f2af9260c876fb722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_81a2128fffdf09353a56661c9e287375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a9638b1193cf74d5d951010c0829f7d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6bbc5e3cc2c16b43b1c83f6b2e7100af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f2fbe32168274b2ef5e4e59e461a0750(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_46a4ce3f9ec6eea8ce02db000b5adf8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_11aa531644efea716fd19d021e4ace1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6a1fc1d8f40c80923ed14416f98faecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e20fb5babeb1e25ffd0d35fab63c0c83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8dd45eaf58a2e05d2e69522ee1a43f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c73259d6a12f54a325161c031f94a030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a9103388ebb754409e5d54292feba9d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d21492bce7df6e56aa247eca40761872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d453ad6f585b4e541e991912bb43a112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3304fc0fec9d69a5baed031f7db7f31b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0bfbb9be65312a6bbaba5f3f92da3efc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c20ffdbf6f3686d37f40aef24d0b5571(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_793bd0b568337c8aa1a078f215a2e160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_044881b14741eb8e0d27c9ce293a2848(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_83324dcf1f10f46ea3a2ebdc394c2a32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0e6f91acb3679ad513a96e35d9203a71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_af3328e0322b99e4cb2c24bdcf49ea16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_91e047ee9c5c05922ca8227eac5febd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_93d21e78e3bb0602fb337f37a871620c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f73c23fa89b5b50d8632a10861d9f4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd5a69fdf2f2cb3f375dd5c2207dce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_703e28b817a16642705e28ed8777d655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0213cd7f3af00929710d9b6f795e2cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cf86df809910615b033b74b317429dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a91f1710ebd45e6d7dc2e352edac8ffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_174f823d63d38a36e6b2bf451edcb943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0f48203981939406f3b04f3346ecb6ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8a2ee94292351a17ff3e2ec867ec0166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b3960ea83ec0a94fe798c231a8ab21ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_084928836c07f632a2bdf51823579a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c07e11528b783cfce461595e19b55248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d72ebbe7fd9102a7284b94e4dfe39322(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_37c6ff986a10361016082c8fb28a1b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_053451d4793e319813358682b73fe0ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30e1dc391bc3960cdca0f7ba902ec45
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_abdaaa0dda8cfdad02984ba76c3deec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30e1dc391bc3960cdca0f7ba902ec45
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_446b4bf9724d5bae7c27bf93f1ccef0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b30e1dc391bc3960cdca0f7ba902ec45
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_346f04adbd12522cf593b962f969532f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1ce2f7b719bfa9e5d606f4e682fcec79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_069cddc67a948d3aa376fc90cacc2427(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5b687eefd27672288658e3c953cc0ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ba7dbf6d23e7c7430f8fb6ba28f00138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_20934c9327c2faf8ff5f2697ce90e239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eba25f82f1398d9f8d8c709a4abf29d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_57751eb33f870eb2816478e982154c98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2289a956ee62cb63fd7043c268b498e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b649611e00f59d7a76753bdb2d13def2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d78bda2f3829b03a047450b1ed4c1219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1102a57102e8114bbe302b13200845a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e81f56ddfef9a13f2f6f2da520202a29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_96947f3b6691d2720a64e8dca10a9af7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_67f92489538c00390f1bc12f8d01d012(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_acd6e5461192750947a6858bff6510e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b02de876e266ef6ec7474ea34e675b66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3774b5568dd372096cac3ba4378adbe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1dfb031ef9ba73f560e6ee1b76e5ff1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fa41881b1aebf95736f15d1de75d27be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cf9f85255f5ac100cdc3611a9f684192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ef426316ddc33860dcbc2b67c5d167b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fa2e2ead937bd8fa45509aae50786874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d2f526b147850f091985b012e30381ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a1790c2d9517480a4fab343a493916d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_15c2ea15fe03dbdc65728a9c1b9f99a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_60ca19ceb07fa67c6e0a5fe2f165f546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f674eb93216b66b624fcc94920cdf8e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ac806fa8184bf65685008c049362bddd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f0a89018db97b1b136e3b599f046bf28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d7b4f867ab34ae9eeea2efd36cf3826c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_780adc3da919a04fce3d5c17b85be1e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1df1b83cb6498c02858c8e1305959d3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1db66a40b0f522dfcec8b94fbdbf26ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4aef48db78deece9c1f3fa8b7cd7c2fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 704, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4893ad1c926f2fd025960d7d99c66473(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9e565d8ef391a7804f5aac9467a9d754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e893a8df3330516975ca89bac085ee67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c08d4c95ae42080b1bcc1548d87f01cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a84c371f40f7cd7b99e00ee3391e1e90
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d17c161a62a64b514534aa2b7a4dd083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66523e9c2b8ecd41591a52d5ea9faed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d05a4b5fc458b44aee937191654c3e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a9c52c83fc8dcb040a7998fc7efa918
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cbac96df6a88e6f381f0ae70bf593469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1551dc93a7b06f99265af625504738f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8f78325c5a24da682d309241ba123eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb27a18c852ef371fa896a7617186027
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]




if __name__ == '__main__':
    unittest.main()