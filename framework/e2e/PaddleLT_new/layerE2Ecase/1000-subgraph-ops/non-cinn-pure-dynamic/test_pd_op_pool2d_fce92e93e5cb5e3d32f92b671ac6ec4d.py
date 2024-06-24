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



class PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b325339bfd317b090c5a4396a6036134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4115006f9e89d23fd34d347c5c7fb88b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9ab33fa14eb4a43c1ab61b18758ec700(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c60cab27de90169aa6f087b2bcd415d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d91b4b97e444b25b2c3c9bda45f2f389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cff18604599a848db36941b2c90c239f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7ef4be6615cbae062b50b176b5f1f40a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9788c382760108c364fd8d0aba356207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_54b7934fc78dc74d39ee995eec7a4d76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2a0a828c95b1b7ad451332db1da45fa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_325eb69ca6c0970385a5cc392ffacf3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ae9d183b612c41f511160504f9720dfb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb856a9094771a777050e6f9ab2a592a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_262fd8833014c159789a068aa2168314(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e00133241ffe1e9c002a9941df201d0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_480088d7ba993f9af96d3f4d3ed16d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1c7751ae570b70200d693eb13ae3301b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a1605fd83adb8bcd0dc3f51406b89cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a609954ba5403e682f284c9319cde265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_550855de3a3b3aede132401aa6e7d61f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e12b9a859d34e4c064143818c98042c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_31dde51b6fa21502c93bc11212f29579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_517534e1ce65a6a325fa55217419345f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_13d33e7242743b56a7509f6de5cf22dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6f03ddf8a593f89e84892cf1ef275d44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b62be98d0c3db836a7d1d45f2125f971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b67091d8af80980f5402724b6accee0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1fcce2d930ec1b6d57fc5bf992ab9ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_92e21b134dad091f18aa35155d2545e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bea1b86b7f5e9444035e226d93a9e839(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_86252b72fd721d6a77553ed16149c391(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_743078adaaebddb385daf13c0900f629(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66d9027632404dc2bd60d3d47ba0c046(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c032e0a9022faeab6062c0b8d6541ddb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ae7fd2f6ef8d1d7aceb6cf5721e67036(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c561392cb8bd0f6f2608d23673a9949b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 2048, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_568ee51d492dc9c1a73ed3b065f11905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ab2959ef4df2b080c1d4ce8b831c8c65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2356ceef8ef4712329fa741ad1bd20a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fec7834596c359cc78e32feecf6f731c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6749f3550e967e07c3e6737f15643d3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0cc5e9a1b9709e0c641de82233a58e10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1024, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_df485639c6f4e7064bb9949926cfaa6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f0e5eacf73c14682523b2185a0558a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e8385b3017300c494dffd87d4de3be34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8b6eee320548ca35f73ead75aee3de12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a3c52cd7456c3022b75a53185ab0a021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d5b6a7e98af8763ef9f96dfa57688881(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_20419186b622176d134f855439d8b73e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf01cd02459be2b4afeef1838dae39fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20419186b622176d134f855439d8b73e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_16f6618af7bf24eaa9ebe8c06b6e1af2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40f1b4df100f1912d58ee768734a179c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f6618af7bf24eaa9ebe8c06b6e1af2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_0c3e1d0576708afd33e6f8e7303f1888(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_893f57ddb0bd0d3c9e8d50c3d42daa41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c3e1d0576708afd33e6f8e7303f1888
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_48a0d5d244daf2ad1cf50fe860105cfe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_921ebf9a27325589f500491856174d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a0d5d244daf2ad1cf50fe860105cfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fdc1d2253d0c9fac99f9eb6459bcdf0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c588940c993f9f42ae54db2f65d8c5b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_86dc1d973e0c391f31fb8495315970ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_625c8b6f636498e68dd09fa24d170fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_59d4d2c24a49a8e5fe92cbdb936642a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f64590030764a084948a1133c1389872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a0d5d244daf2ad1cf50fe860105cfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b440d3218a77133b4ef331e38297911f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c3e1d0576708afd33e6f8e7303f1888
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3470a9a2c679b20a5660b5f296d1d6f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f6618af7bf24eaa9ebe8c06b6e1af2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_01a2589ec00d26ef05d550dc83865783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20419186b622176d134f855439d8b73e
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4b37998f0008f7aefc4581afa3b570c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a0d5d244daf2ad1cf50fe860105cfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3409c1dd823135613e07666cf2bc5232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c3e1d0576708afd33e6f8e7303f1888
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_375b1d7650b05a0518711b64c0fd5c15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f6618af7bf24eaa9ebe8c06b6e1af2
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0932bcdfed29d1259a1737da21342231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20419186b622176d134f855439d8b73e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3fae07143bbd92244021076fcc08da22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8b81259e73373d964a3ee2046dd03591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7ef4be6615cbae062b50b176b5f1f40a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ae13c4deda88df5eb4ee82b92e0a317c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_df484c46e19cc78d33b6a21da53ee520(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8eb32e29f41dae7a3d0079c086c26b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2ea81e7475002df9872db5f5649cce6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3ba69708c0de6855f5677200574a827e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_30b9ec5b6fda80267f37ea28b4a0dc58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_80253729f76101ce8c7148ae285c5dae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0df579fe52e9b514614f63abb2c01078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ad8d209bf926c46d98678e679c9c6ca5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b6eeb730656f186bc23ec6f51a6ccf75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_545746cb22cf0c1a47b71948a43f2650(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d76f9195b7463de7e96ea99373d21f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0328db8db94bf18c0c692e3011fd5375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_03d8cebf8e83bd3eb4f86100e298787e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fd53d32eb4c3344055fbf75690aba5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_53aabde2a476ab5562c9cd8f5e82dabd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_18bedba1c1f38e75310b0212d6039ed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_25f46c497b20b09521119d05a33caebb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1cdfd0f1afa1f2d54bd36cc1b3a9f1fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5ae0a5559af5c6e6a2fc25ca933043a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7d46c4864c5f1548b3ce6b8a94f83a06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_71f4fc663a218812f85e4d1bf4225b13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b9589802b7a4a373c2067d76070287d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ffd0c7a68b846b0bd4e3624c8b884479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_83ab57cee2579a71415cefa49e5c3934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_97353849b28232a48d6ce61c07dd4926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7af0c62945a59f31012cdf007a63c5e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b8379947047ab98bef20efb3bb30f9c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b98d940dc8450287606f919996c1a06b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2e0810273b6fe30a8b145c2c92297db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 320, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_815258cc22e9a0a5b7cc6c366b78184d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3c1dbed67ba073e94d0259723abf957f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1b1e4d6f23f563842d4bc2a6864062d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e09895756855d09faf4b9f903a5c7fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([43, 512, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_be821e532ccd6fbb6319a11a161de91a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_452289bfaf7b0e3ec1593d42acd9f90b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e9c6ffd062b14010f482c76d556b6a8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_befb8f3aa5b325180586aab82fd7736f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0f0f1813d7d86acffd39dc45d68fdd46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_79e492996609a4df8053b7f9c7313176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9e904761ac412636566f813dcc0bb0ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b02af7220505bddbc577142b771e86a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8bfb8b65b67ac4e18ebf54f037335e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 2048, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_655e6f4fff5b54a31410303e032e6361(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4d022a0ca9b62a0cc11879fb2d068b97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7526b59a647f57de47c807b91797966c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b67091d8af80980f5402724b6accee0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0fb5801a59a4d39f01e79ffa04269aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0f1e24d2822b851f6d0171d3967b994f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9465a2c557980a8421a3e570fdcf82ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d2c849bf2e18f927e15bc8356d8ebb0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_841308a7b7a6378c069a21ee661f895a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_72fa8b966f7129e0577d7090ec8366d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0c8417cfd6b19001aa57e9def2624f7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4f23a1244b3ac76a131365039492279a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b2070f9507d175e3c3c81d64b336aefa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0def6f2cb18b91a0c1c397632b98c4e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0def6f2cb18b91a0c1c397632b98c4e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_472c73177d44df9289f6719a94e028af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_472c73177d44df9289f6719a94e028af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7c318ea470ce3365fdf5edde5d634eb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7c318ea470ce3365fdf5edde5d634eb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a82b438101c61c2495ebec0960e6d19e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a82b438101c61c2495ebec0960e6d19e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5ee19170f2ac9bafe53ab401e5c843eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eb6c9af13f84b7208480a805573636a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eb6c9af13f84b7208480a805573636a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_065683250a8efc095a3bec8e2839c9e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_065683250a8efc095a3bec8e2839c9e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a02a664aac9cfeddf7e2a87c3d8e8281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a02a664aac9cfeddf7e2a87c3d8e8281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d8adce20c3d735451840d3024b5d0a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d8adce20c3d735451840d3024b5d0a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3487737203c836ab3eafab565bb04dab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cd3324b2d52970eb12f80b2d1a4b1a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_97353849b28232a48d6ce61c07dd4926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b61ea4ef5325118f40511c7ef4a734ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_041efb64a2f7033a54256de7fb4bb6f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_03664ef0a51730d52c2484c75d9eec4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4976588342de53d195ac6ecf5c5aba71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_aa94783c5e47e1435e588b30ddfb0d6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_745bcce115b751969dc2032fece5342b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_82f95a97d7d2389593ff5b2ef650060c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_82f95a97d7d2389593ff5b2ef650060c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dc8b17507e3c3f7815ffdd0b0e0dcaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dc8b17507e3c3f7815ffdd0b0e0dcaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0ec8c8ecbfecb72b2983ae70911ef3f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0ec8c8ecbfecb72b2983ae70911ef3f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bc889725b1526db10638b4290479dcc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bc889725b1526db10638b4290479dcc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1f4b755751f58c1ffd3997526f0e286f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b171226fbf8e53bdf9bc7736d8a4e698(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f4b755751f58c1ffd3997526f0e286f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c7b434e43bd4fea29b90642f6d4a9b2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f4b755751f58c1ffd3997526f0e286f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ba36cb733207760b9756b283cacb21b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f4b755751f58c1ffd3997526f0e286f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6a0a27cc97a4508dd932d7c62b7bc7a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f4b755751f58c1ffd3997526f0e286f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_5371b647b256f531912bec9455c302b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [1, 1], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97e7877f9db92488c50348f098a099f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5371b647b256f531912bec9455c302b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f61437b958057816912b411e69a53c24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_948f87a49d929457e40232f6dd1e1200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9179ecf69d514125b2b641ec94bbd1ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_27894b46f95b7a5b9b3b974ccef7a1d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20419186b622176d134f855439d8b73e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3604039447752ba06cebb050a757d5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f6618af7bf24eaa9ebe8c06b6e1af2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eb23c8e0a74bc9d33113930dee3950c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c3e1d0576708afd33e6f8e7303f1888
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_be96068dacc56635249edd6b1c433703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a0d5d244daf2ad1cf50fe860105cfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_adbb1deedbe4af5633eb9474c6cd50fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ff81adf83cd09675a6685af6b9d21fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fc27dcc0795e7abc245b0a91c2303449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_396051423a2bc1e24a17fb6cc396ce8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_629f0cccb66f2adc495191aef0237106(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_62f8cccfd2663d34808900742febb8a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NHWC', 'avg', False, True, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a62134d53777654a83a474f19c0fe70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62f8cccfd2663d34808900742febb8a8
    def get_inputs(self):
        return [
            paddle.uniform([22, 7, 7, 2048], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c47e4eced5411f75a5ccc93a7d5ae1c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_af966670c0d3c987bdda7009408fe2b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_456530d44d46cddac00b8bfc95c4d4f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d9cbb5ef21899b57f5edfa0f05ae18d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_585b6b0a198e0bd60a36643b344500b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5aa64ca67ba50325ca1f37427a9ad21c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_88f702fbb9167f45561975cd41d3a018(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dfc034cfec55f7a6ec1908febd231905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5ee19170f2ac9bafe53ab401e5c843eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bef8a1afd6aa4525dfbdaec7f97d9ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e24d902edacdcf7b5d01811480c08871(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b72d795e84250e28d7ffc7f75f609870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2f9cd82c21d4687245996a5d93a3d145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_93d5bbe56e58d75cb154347938275e65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_81bd936357ed00968001d18952beed79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d673835bbd744bee89c55a90fbc4f749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cec1bf266124acc5f192a818fbf69a66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_97179b70dd4565d9e9c03fd6ae5cba8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_398fa1053248ab41270a0a056b21e19f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f2b506f06d547121438700bfafe36004(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([11, 512, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e37ffb532d742a9b0cc086bab8f322c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_169bd13b76c82831cfcb24492fbc05b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_475570f4e60a077627a73a38694207a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_669e5694e22cc225e49fa3c4596b2b05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9153178a39657248fae2b121f86edacc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_91fccbe6f611aadcb59bfe35c6658972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2023d51256c751f61ef9386434382aa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ab6d4d4d7dac0e4b9e61d9e655050325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c4d5b74df56a9240b45d196aedd7a699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1439c4122c11c33268653214e0f5573a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5abc3de2eeb7cd4f67c38eb3dfca1db6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_877c7989dcb51b76fd3e7fc5b5facb37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_72d9ea9922493a758e7fb83058c7a1f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_390f9f7b64cc0ae98b73752fd3736439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9e5af1f782e963632a2a9a0718d25d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_853b5bc46d67dfef99c497f3155cd794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20419186b622176d134f855439d8b73e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_78dd66ce985789fb62a29dea538fce98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2d5ec16ab16c044d3e94464d718ae45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78dd66ce985789fb62a29dea538fce98
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_65500326b0a443807f25556cb477f27a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1e2375d8fcb02f4e91691ef1d8fa7dbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a13efec23a2c35b24b4174841d072201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_00199ea71722a5c90f3a3810b0270f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f1d95f5622330c6af74faa52e31ff78b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ad92b8b91e5095d4793456eaa187fc95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8b35e2be48223a188a7a87ede51de224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 320, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2fef1bc32e705b8ffa985b434317f538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2a8eb1aeede9b61958e438ba21dbe5fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f4d1379f43d4a208d7b0f73fee88c4d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_14fb25aadd3fd007fc7afa80d8c3d159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fef3268e6ca0997cdb1efce14318751d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_721e79640b22475a6a938d8dc31f9e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_039dceca3b271bfc52e7e45139bc6d65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d061925bb1e5c89038818b1cfd6c1f53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1024, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8d1de560e268c11107072d8cd0fe8801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a82203f720a54dd3a312674a424b53c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ab6d4d4d7dac0e4b9e61d9e655050325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b8222c9fe54d7dd112cd4787f4d15cce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78dd66ce985789fb62a29dea538fce98
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_55b833519baa9d2503b399bd73af09dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e7009da49ff462cf18163103a46fa699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_731a07f78acf74f8cb7a5321bdd551d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b02af7220505bddbc577142b771e86a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cb11f0bc8f0b0f4c145146d3e497fa3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fbd978ccbb61c3d3d70d8a17d55e40dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ca9a4d0928c3910a83c690e68e3d40f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0240ae8ba3a2ca5b92e4aef73ae64167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e48f65340ec2365b28eb1feaac9d59f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ddbc620e26b31bbbd05a4813ceb71913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1d7851d30adf1119f05237e3150deab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_560c054403ebcbe12cb6f52fc4fbe30d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_041efb64a2f7033a54256de7fb4bb6f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8f3248ab0863ed169f3923aeb69030d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2f9cd82c21d4687245996a5d93a3d145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_68c2e98a987e277a150ec6225fae8cb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3fee1455adaa5830ce80143508a9fbb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_17d336a2db1a7eaa815a69ba127395a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4633cdd4456fbf80c5816aa28fd8cf21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a3d906d069cae157a109bea9f0b82708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5369269b6e517011b321b806085ebaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bb656d71ada64aee6ac45c1933076ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8f5e1516828650cfdffb305663777199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3e25d8ba8ab04c99d71ef22d84753edb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a3a0b8d34a062188ec5ecd503035e3af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6d490e85e77065a0fc1325a723f0ee97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_13d33e7242743b56a7509f6de5cf22dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_3f73e16a772cb40c9347b5c4a0e91c60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.pool2d(input_0, input_1, [7, 7], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0facf519eed45f539fb55039cb323594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f73e16a772cb40c9347b5c4a0e91c60
    def get_inputs(self):
        return [
            paddle.uniform([11, 704, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_31dde51b6fa21502c93bc11212f29579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eeee697fec0b9ce2a28fe4673601aaf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d10c7be51492017b541595c8a505be16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cf10d2809216e1af60115811ce849dfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_073f128f4fa69bb7abbb4260144028f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b96a6c64f952e6a46db312848e07ca92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_77b532b57cc02bfe2363a689f7d54007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_77b532b57cc02bfe2363a689f7d54007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bcab2c3d89a91f77d45abc22a63a7ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bcab2c3d89a91f77d45abc22a63a7ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a8b174ae7b78ff63d31ed720697d84f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a8b174ae7b78ff63d31ed720697d84f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a2140d9604809f77a34686c86b986ba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a2140d9604809f77a34686c86b986ba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3618f8c14431fc0912e06174de76dea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6e21e9dbeeb9bc7fd7141c6351970c44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6c23025faacdc13d12c0768f62507677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3702e649c83aab16655ae521227fd137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b0f7ef821872fc45f604ff42effae904(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_38b5627bb2f9b9b2a9d1eea1fe7971d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4a168fd9893e6fc810d5009b2bfc8b21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e3c2e1f937905857d698f5f8e3ab5149(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d267d69dbc2f910c8f23a15fb4ecb256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7f111bc25ea673928aa85e7359001598(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a0d5d244daf2ad1cf50fe860105cfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ec1fb7dcac0101c8bd3242753fe426be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c3e1d0576708afd33e6f8e7303f1888
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e2b60c77ddc25b0fd327faceb6edf93a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f6618af7bf24eaa9ebe8c06b6e1af2
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ad97e810f2f1f7ef4d14ab533fe59047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20419186b622176d134f855439d8b73e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a1605fd83adb8bcd0dc3f51406b89cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a609954ba5403e682f284c9319cde265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d21597d6865ec2c1602208c9312f95a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5b2499466102de214c1f6a5f8225d18b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3157ed739b3c106bc1cc84bcb56da91d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_73391d94dac87d4231d62bc6e5b0fdee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4507b11a6903f83063061520621e992f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_21f62b176100850b4161461283355cf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6ac78707e5cc7612cd88d4543043ce27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f4f2d356dfa3a1ec4cf1699ecb9dc68d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_34dc4709bc787c9f4e3bcf251ccffda5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b1166c07497130efb71f515d8f58f181(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_58aa5f45b76ae90659f28260efd9bd5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ab6d4d4d7dac0e4b9e61d9e655050325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_36aa69ce62c9df5458f2f29916ec1f83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5994d99aa843bdcde8cca7141e8e0edf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ef31e83eb96f1bb6eb76d3ba602194c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b8c872ff89c7fb703ed1417d7320b09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8306144e92ea7193f312c91dd6bceb7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b9e765d8fc768f76efaec1cf1d50353b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a7eda13eb39785a2fb5fb6880c7a82a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c993a546e4b4a0175242c2055127c02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_03eccf68be23f2b644d85cac6a02f49b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_05aa4edea2477217c896a879e31b8f20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_88a2b46d142ad5fc78ed67b3f8920681(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6093cbfbe853c1b1eccba47a48eda647(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ab73ca596cd239a872214e8f8e99a39c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a1605fd83adb8bcd0dc3f51406b89cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a609954ba5403e682f284c9319cde265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d21597d6865ec2c1602208c9312f95a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4507b11a6903f83063061520621e992f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bc39f60957d01d94f00f70c00b073d78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6ac78707e5cc7612cd88d4543043ce27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f4d1379f43d4a208d7b0f73fee88c4d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1f3b2ed6e985be32fb2c940b45f44922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7cb12eaf9fdeb5058424d347eeb0d47f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_937297f6a3aeaf801be3ea2002d9f233(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_41256d08bb551149d1b0d8776ba90ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a8857cc50ec1f25b9603507f4d7fe157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9e3f26d82f6ec3f2ace66fd7592d3c46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3a8f9cc32424596ebc6a3c6f2615f187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_17a0860cac790badf6348cdb5ed7aaa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cb80a3e9215fd88bacf6198fd17a2086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8a1220abcf15b15ffb79f11f16a1617d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_812136a4bf9b932486adcd3d52a38519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_89655b176fdae04319f51cc13bc14ef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_45d2fd8cb7c6b1212846814145a23b35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9293d407224788e3a6371e368f2775b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64205d010d19793f625d0cc9c3c42f64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0a056033c52472c0594dee05384c7843(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a23b8c8f67d2371b0db0f2cdac33d57b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_745bcce115b751969dc2032fece5342b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b371918d159f9f1ad2b7fc18228c34b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9df13c1eb9160d28fd162c1fd36b5344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b07e9e01da923e5536893238cd973438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1327ffbee79501d3d798c6cd6ca73628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_71a9215d9a72961d1c0436154950e651(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4337b26e81a42271907d17221d31dbe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f9dfe725db0b50e5db883ee0e47b91b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_837feb974d21e222244587fade1af87f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743078adaaebddb385daf13c0900f629
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_31c7919a8a6341f79dbc08d9e1b0b2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ac5b20466b8c65db1f1c0e327c98e03a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f5752c5817c2b9bd78c8965db0fa1ea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6359bafb3dc2686e9a9cc4c890a8ffdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_06f725fdacea842e70fdd6333c68ef83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c377cce6c172cd7203d16a3a58770135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2a8eb1aeede9b61958e438ba21dbe5fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4f59530a45d538c8ba2ccdbbb17a8ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c9633e9aa6cbf689d5762e3460aee11c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ffd0c7a68b846b0bd4e3624c8b884479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_feb21aaaaa290344af7d4adc1ad5da3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_03664ef0a51730d52c2484c75d9eec4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_df484c46e19cc78d33b6a21da53ee520(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_34bcfdf8e2ee5644bb238a9b4a098ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fa6211982b72c89d3cf6dc01e7899b12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_60830faf2df8152b4d1b744c48fea738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5abc3de2eeb7cd4f67c38eb3dfca1db6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_15f5734835f9bea429e6ca69fb134b60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ad98c1bf8490ed37e888537dfed04153(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6e21e9dbeeb9bc7fd7141c6351970c44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_afe74ddec043c9897341f7bceffab57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c1c50e2463fa07cce460c5b5ee68c4bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04be7338724285c2041b76b0dc7d154a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_27672bf3828213b07c2a4de55808e71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e12ee5de0fd0aff12dc7e65fd27c7d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ac0ada2148f1d1375f6f7ab9e1af6637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4ee8984e544388bbd6ccd8564300f239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_aeb008e54abee3463d31dfe36091837e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f334a5e45d1028c1f1c9877f1705a72d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8e8e570d9d86359049d38829b8d1ce9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6d3b3442f6d50a2605c7f03abf26ce68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b2070f9507d175e3c3c81d64b336aefa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0b4f9bddda473b91f17b6d2e5914a9f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a78d2b1bb5b18887480e9eb6a58d785e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f73e16a772cb40c9347b5c4a0e91c60
    def get_inputs(self):
        return [
            paddle.uniform([43, 704, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f7fa882fca9a3ef07033b72bd2bd3127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_013dba4eb4c87670219ef0ae9972289c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_06a41e79cbd9b6e7584b195f8046886c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4e92025553af7f4ec42763564508a6d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9d183b612c41f511160504f9720dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1e8f1875e1970f2555f39fdd23e841b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262fd8833014c159789a068aa2168314
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9fe4895863e4cfb816d8a3939f991d21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ea3a03bfd2c805ad983f67115506d8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5a2b387b018adb9a6c814365091c4e08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1fcce2d930ec1b6d57fc5bf992ab9ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_88a9fa1cfe3775463f56312d6382dacd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88339cf4a93bfcdd16622d4d09abfc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]




if __name__ == '__main__':
    unittest.main()