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



class PrimitiveOp_b7f6e5ab400411ddf5e93e9f8fa5b858(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74e1930d61a5c79ca8663ec3c21e7eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7f6e5ab400411ddf5e93e9f8fa5b858
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([300], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_579a19a8ff5390566425a8cb8733365b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98ed42d1280709412dd8b4db2413a764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_579a19a8ff5390566425a8cb8733365b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_581d71ab3abcdc75a43ebbe772f31843(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b6c8c8005bd6c218defd82f467b2f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_581d71ab3abcdc75a43ebbe772f31843
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_6c6b72b4b98ebc8abdeeb1f8937badcd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_661742e9d48fddd881be275054f6b0d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c6b72b4b98ebc8abdeeb1f8937badcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db484f0bfff207bec24415e84170ee7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_38881b4e73ea05bb30f4c50c0c4bea50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f3e2490afb6d26a304328a6cd0c30d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38881b4e73ea05bb30f4c50c0c4bea50
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_b6acd042b8ffe13e047254badac2ee81(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4cac7840f2d4a3bdb8ca5d326115475e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6acd042b8ffe13e047254badac2ee81
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_4bd7b3214c859bd28b792c470b8b7dca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f6198a286569bb2cb3ba209a304feb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd7b3214c859bd28b792c470b8b7dca
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_a5da554bc3fca8434943d17c9652671c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 168, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_020d07e7ea2ed18ea8f7bd0d12449f54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5da554bc3fca8434943d17c9652671c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.15382884442806244, 0.28971046209335327, 0.08003828674554825, 0.12820278108119965], [0.28591519594192505, 0.2164941430091858, 0.33096110820770264, 0.17948289215564728]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_04f65dcf70ec8fc7e3348b8c4bc6bae3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 84, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4cb24dcc45b53dc727edcba3bc3cdcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04f65dcf70ec8fc7e3348b8c4bc6bae3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_30457bf266a67f1435fbc402df69ece2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 42, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b35c403aca80bcae8e04794ec534057a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30457bf266a67f1435fbc402df69ece2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_3b0fd77755f08ae13dc317f1bab91aa6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 21, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e920f2ede264fa7f0fd28ee5ba72fbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b0fd77755f08ae13dc317f1bab91aa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_34777eb88d2fd0a1218c57dc83a586cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7f6e5ab400411ddf5e93e9f8fa5b858
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([100], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_79ee44fad721f69356255c5701e83b2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_579a19a8ff5390566425a8cb8733365b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_898ce87af634164d15824f613cb3de82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_581d71ab3abcdc75a43ebbe772f31843
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8173d8bdb7b7310844dc9e586e35d51d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c6b72b4b98ebc8abdeeb1f8937badcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e4af30c2d95c3357df029d0d4d3c106a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.13181999325752258, 0.017232388257980347, 0.4576629102230072, 0.06252817064523697], [0.12928973138332367, 0.38516196608543396, 0.1747119426727295, 0.2775791883468628]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_7e53106752310bb61dbb9f1a836ee206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38881b4e73ea05bb30f4c50c0c4bea50
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c4f95e670e7229562fdba82a1f8148e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6acd042b8ffe13e047254badac2ee81
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_027e38450c4c0243e909508a53983350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd7b3214c859bd28b792c470b8b7dca
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_10d7f9cdbfcce2d8b3982046448f10ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 200, 304], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ac08b7d577d140952cfffa823800ec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10d7f9cdbfcce2d8b3982046448f10ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.0029426408000290394, 0.331410676240921, 0.33316874504089355, 0.36570319533348083], [0.2710677683353424, 0.46047285199165344, 0.15610994398593903, 0.006034566089510918]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_b26e5699e4dcb9da1b51aa4d5d09cbc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6805a4083d295de32bd3f9b3f0018431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b26e5699e4dcb9da1b51aa4d5d09cbc4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_9b1a4905f56db3de3c458b5601c821ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a75cbc40674bf52049bcf9ac5f7012e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b1a4905f56db3de3c458b5601c821ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_5656c6de7d7962c01c6fe6fb3166674f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85178f48dead4e885e0442dc784accd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5656c6de7d7962c01c6fe6fb3166674f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_db484f0bfff207bec24415e84170ee7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2f3e2490afb6d26a304328a6cd0c30d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38881b4e73ea05bb30f4c50c0c4bea50
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4cac7840f2d4a3bdb8ca5d326115475e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6acd042b8ffe13e047254badac2ee81
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_0f6198a286569bb2cb3ba209a304feb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd7b3214c859bd28b792c470b8b7dca
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_1d1b55995d05525b6c1230e7e12dde30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.42700356245040894, 0.14318107068538666, 0.3704715967178345, 0.2660079598426819], [0.28465038537979126, 0.336334764957428, 0.4981398582458496, 0.4699998199939728], [0.14491009712219238, 0.27180197834968567, 0.43630558252334595, 0.1484665423631668], [0.4276387691497803, 0.46587538719177246, 0.2321729212999344, 0.09956532716751099], [0.3571280837059021, 0.005369572900235653, 0.4875163733959198, 0.49724993109703064], [0.03416522964835167, 0.0690198615193367, 0.10645696520805359, 0.4639917016029358], [0.1267307996749878, 0.1599087417125702, 0.18144196271896362, 0.1197667121887207]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_41e5c7eb9a63777a739266a7537fc825(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38881b4e73ea05bb30f4c50c0c4bea50
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b8a6cb4bfe5fb03a71fb81afbd69d4f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6acd042b8ffe13e047254badac2ee81
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b57bc9f8531e91177598087484abe6d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd7b3214c859bd28b792c470b8b7dca
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_596ac55d96e3b18170342caa7054fdef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_475fa9fbfd8a9dba0c5315cd2c4b1b3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.13693512976169586, 0.330445259809494, 0.26659926772117615, 0.32291409373283386], [0.4757486879825592, 0.10236584395170212, 0.06810323148965836, 0.32069897651672363], [0.07968559861183167, 0.0039863791316747665, 0.289151668548584, 0.0654677301645279], [0.1676672250032425, 0.3673466145992279, 0.12791667878627777, 0.320968359708786], [0.30439913272857666, 0.018727809190750122, 0.18651922047138214, 0.10556644201278687], [0.33935245871543884, 0.34426581859588623, 0.019318874925374985, 0.009654301218688488]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([6], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_93eb03926e80aaf89790fd0d70d3f4b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5869df3b9e13a5650b6b8a0c89afd593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93eb03926e80aaf89790fd0d70d3f4b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_5e130c493186dbf2a2f48cd0515ede4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8b8ee7922e70f669831d91f76748e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e130c493186dbf2a2f48cd0515ede4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_6e04cea6c0fbaa9c78ef4f5bbbbb37c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fbca39b5e0cc7f3ce8efc8e654747bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e04cea6c0fbaa9c78ef4f5bbbbb37c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6d19b586c67d5ec2b72715552bdc3019(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.025969570502638817, 0.20490652322769165, 0.13725070655345917, 0.12557759881019592], [0.4437328279018402, 0.4999356269836426, 0.46197038888931274, 0.4470215439796448], [0.07791996002197266, 0.19307906925678253, 0.3600310981273651, 0.485014945268631]], dtype='float32').reshape([3, 4]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9ee3f45635d8ccf4896bd98b8992a0ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38881b4e73ea05bb30f4c50c0c4bea50
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_125d8a1873c6858c15cec12c1986c024(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6acd042b8ffe13e047254badac2ee81
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a1b16c9ccf6632ed146df37b1166f211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd7b3214c859bd28b792c470b8b7dca
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_1f89a876bc7e14ac2538f91f1ef55a76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10d7f9cdbfcce2d8b3982046448f10ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.10802742838859558, 0.18062922358512878, 0.009967976249754429, 0.4450889527797699], [0.3360653817653656, 0.3106331527233124, 0.18778148293495178, 0.3714531362056732]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6805a4083d295de32bd3f9b3f0018431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b26e5699e4dcb9da1b51aa4d5d09cbc4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8a75cbc40674bf52049bcf9ac5f7012e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b1a4905f56db3de3c458b5601c821ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_85178f48dead4e885e0442dc784accd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5656c6de7d7962c01c6fe6fb3166674f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_95bcea33de51c167ea8f1d62f8398f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.3556005656719208, 0.3736079931259155, 0.2928246557712555, 0.25168508291244507]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ad55397e73d632759a1419b176ad70a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93eb03926e80aaf89790fd0d70d3f4b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b8d5dc264570c8b67df30d8045d71c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e130c493186dbf2a2f48cd0515ede4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ffaa199627160097224077a4a8627b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e04cea6c0fbaa9c78ef4f5bbbbb37c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_eaaee57db846a8eb8364a7f0c170d275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.13698779046535492, 0.4151155352592468, 0.15837134420871735, 0.4031403064727783], [0.2384192794561386, 0.19400882720947266, 0.3161297142505646, 0.4227304756641388], [0.3381301164627075, 0.10545167326927185, 0.20200076699256897, 0.09810540825128555], [0.4387856721878052, 0.18500865995883942, 0.4693276286125183, 0.379557341337204], [0.2603653371334076, 0.25184184312820435, 0.10041908174753189, 0.3186939060688019], [0.32229602336883545, 0.4989703595638275, 0.06026025488972664, 0.46269917488098145], [0.36243900656700134, 0.034315627068281174, 0.2633066773414612, 0.13660143315792084]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_fede7e3db48d572b2e689eef1016ebb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38881b4e73ea05bb30f4c50c0c4bea50
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_126462afe2714bf0d76052847b86752c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6acd042b8ffe13e047254badac2ee81
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c915c86063fdfacb502756b35abf8de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd7b3214c859bd28b792c470b8b7dca
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_20139494cb18ab0de139ea290a31bf91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.1792377233505249, 0.4446202516555786, 0.046680793166160583, 0.45548316836357117]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_5869df3b9e13a5650b6b8a0c89afd593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93eb03926e80aaf89790fd0d70d3f4b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e8b8ee7922e70f669831d91f76748e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e130c493186dbf2a2f48cd0515ede4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_0fbca39b5e0cc7f3ce8efc8e654747bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e04cea6c0fbaa9c78ef4f5bbbbb37c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d525d49b21adb7cb4d319621b58ffcf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.41165491938591003, 0.208968386054039, 0.3744460940361023, 0.34794318675994873], [0.24314932525157928, 0.39855390787124634, 0.21495309472084045, 0.45318329334259033], [0.1065288558602333, 0.1423860341310501, 0.3848417401313782, 0.393140971660614], [0.25039365887641907, 0.40869447588920593, 0.3528030514717102, 0.144731804728508], [0.2517305910587311, 0.2205875664949417, 0.08375892043113708, 0.11715321987867355]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([5], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4f2bdfbddce71e307393a69b3cb47aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38881b4e73ea05bb30f4c50c0c4bea50
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2cda2258557f9ddeba479eace484922e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6acd042b8ffe13e047254badac2ee81
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_0637405dd649ed50272dca61b348efd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd7b3214c859bd28b792c470b8b7dca
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e49daf7bfa54ea05d6e1b2f1ca283de7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.37797877192497253, 0.29004013538360596, 0.20578141510486603, 0.3348413407802582], [0.3425464928150177, 0.2963353395462036, 0.4219314157962799, 0.2853028178215027], [0.20327356457710266, 0.06568564474582672, 0.0010085736867040396, 0.13093438744544983], [0.46062493324279785, 0.2800303101539612, 0.09548783302307129, 0.4964846670627594], [0.4840283989906311, 0.2591872811317444, 0.39890962839126587, 0.3481963574886322], [0.411825567483902, 0.41528668999671936, 0.40591150522232056, 0.09553781151771545], [0.40305840969085693, 0.4324552118778229, 0.2683946192264557, 0.4635322690010071]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3b29552cedc9351c4911174c6f60c48d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38881b4e73ea05bb30f4c50c0c4bea50
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_1035f5f58970b538253ffbe90c142850(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6acd042b8ffe13e047254badac2ee81
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_539e2b0182cfe41e2cc9b2a3fed550a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd7b3214c859bd28b792c470b8b7dca
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_27b5f85f2d7f33078ccb8ce9d391ce76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.1397065967321396, 0.43400922417640686, 0.4426351487636566, 0.024016454815864563], [0.1896745264530182, 0.1307414323091507, 0.016309713944792747, 0.3397108018398285], [0.21210622787475586, 0.3925955593585968, 0.05922114476561546, 0.08246976137161255], [0.21685650944709778, 0.020440155640244484, 0.11512778699398041, 0.4176737666130066], [0.14286744594573975, 0.3935248851776123, 0.46487298607826233, 0.3819267153739929], [0.19045846164226532, 0.3170645833015442, 0.03019396774470806, 0.4296601712703705], [0.4562056064605713, 0.36171892285346985, 0.062002625316381454, 0.1590944230556488]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_41e5c7eb9a63777a739266a7537fc825(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38881b4e73ea05bb30f4c50c0c4bea50
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b8a6cb4bfe5fb03a71fb81afbd69d4f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6acd042b8ffe13e047254badac2ee81
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b57bc9f8531e91177598087484abe6d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd7b3214c859bd28b792c470b8b7dca
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_103ef4d005cc026d099b0c163457018e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.087706558406353, 0.1823846846818924, 0.451652854681015, 0.35312846302986145]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3a40c81d27a16a03bc22f873ee5b88f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93eb03926e80aaf89790fd0d70d3f4b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_297e322b986f4a687a30a8f8d2783de2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e130c493186dbf2a2f48cd0515ede4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f744e73322bf8ea29af756e29ac505f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e04cea6c0fbaa9c78ef4f5bbbbb37c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_74e1930d61a5c79ca8663ec3c21e7eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7f6e5ab400411ddf5e93e9f8fa5b858
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([300], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_98ed42d1280709412dd8b4db2413a764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_579a19a8ff5390566425a8cb8733365b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9b6c8c8005bd6c218defd82f467b2f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_581d71ab3abcdc75a43ebbe772f31843
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_661742e9d48fddd881be275054f6b0d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c6b72b4b98ebc8abdeeb1f8937badcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4dd0109f07d22781f10efccc5946687f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.0009780547115951777, 0.4094560742378235, 0.27823761105537415, 0.2222481369972229], [0.17997577786445618, 0.29044869542121887, 0.4062460660934448, 0.4734513759613037], [0.2983068525791168, 0.1707010716199875, 0.17511555552482605, 0.3212588429450989], [0.4697624444961548, 0.23548531532287598, 0.2719791531562805, 0.38895589113235474], [0.1657763123512268, 0.23838230967521667, 0.26258766651153564, 0.3946164846420288]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([5], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4f2bdfbddce71e307393a69b3cb47aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38881b4e73ea05bb30f4c50c0c4bea50
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2cda2258557f9ddeba479eace484922e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6acd042b8ffe13e047254badac2ee81
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_0637405dd649ed50272dca61b348efd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd7b3214c859bd28b792c470b8b7dca
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d01c324817a499f9af89b2a99d3f7049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.4950709939002991, 0.10976904630661011, 0.34217217564582825, 0.222353994846344]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9d326ffc52805465e08fd198e9b24b04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38881b4e73ea05bb30f4c50c0c4bea50
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_492bc79aff21273ac0cc1a6822b8518a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6acd042b8ffe13e047254badac2ee81
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_801dc7af1ca16e7a8cf23f9b627b377e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd7b3214c859bd28b792c470b8b7dca
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4765726f4fb98644b31c9f109742537f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.0468159019947052, 0.3286653161048889, 0.32383060455322266, 0.2822246849536896]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2d46a54920c1b2ebdd43007399e72306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93eb03926e80aaf89790fd0d70d3f4b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8f50cb7978b705cef43850bee00d61d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e130c493186dbf2a2f48cd0515ede4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a8e7a45c41c9b7fff25b69902feb01c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e04cea6c0fbaa9c78ef4f5bbbbb37c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_7c5e503fb38accfe19b8efaa0228a392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3a40c81d27a16a03bc22f873ee5b88f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93eb03926e80aaf89790fd0d70d3f4b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_297e322b986f4a687a30a8f8d2783de2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e130c493186dbf2a2f48cd0515ede4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f744e73322bf8ea29af756e29ac505f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e04cea6c0fbaa9c78ef4f5bbbbb37c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_34777eb88d2fd0a1218c57dc83a586cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7f6e5ab400411ddf5e93e9f8fa5b858
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([100], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_79ee44fad721f69356255c5701e83b2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_579a19a8ff5390566425a8cb8733365b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_898ce87af634164d15824f613cb3de82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_581d71ab3abcdc75a43ebbe772f31843
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_8173d8bdb7b7310844dc9e586e35d51d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c6b72b4b98ebc8abdeeb1f8937badcd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e4d62b0954c94d46b3fbc14947022203(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.1843632310628891, 0.08669140189886093, 0.11040329188108444, 0.4772191047668457]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_19d8f45c3a1e85ce2e09c9bcf58b1585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93eb03926e80aaf89790fd0d70d3f4b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f305485119de8d0361b48cd41efd9fd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e130c493186dbf2a2f48cd0515ede4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f0c3fa279cae7ba9a4063d832a715587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e04cea6c0fbaa9c78ef4f5bbbbb37c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c3253a6b3a1abd609025ce5dc26c5d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.101911261677742, 0.07098075747489929, 0.04293031990528107, 0.41263023018836975], [0.37657681107521057, 0.22034059464931488, 0.10211452096700668, 0.32544079422950745], [0.43616431951522827, 0.34981438517570496, 0.14842607080936432, 0.44913846254348755], [0.14437170326709747, 0.1833578199148178, 0.4404062032699585, 0.17392417788505554], [0.19155532121658325, 0.48286986351013184, 0.25538915395736694, 0.38385650515556335], [0.3828704357147217, 0.3184468746185303, 0.31152549386024475, 0.1916542798280716], [0.08516884595155716, 0.017141643911600113, 0.23004117608070374, 0.13147026300430298]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3b29552cedc9351c4911174c6f60c48d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38881b4e73ea05bb30f4c50c0c4bea50
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_1035f5f58970b538253ffbe90c142850(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6acd042b8ffe13e047254badac2ee81
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_539e2b0182cfe41e2cc9b2a3fed550a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd7b3214c859bd28b792c470b8b7dca
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_726b71e30ecf2cd5224ef6c285ed8016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.26911649107933044, 0.08277752995491028, 0.36353999376296997, 0.4030780792236328], [0.050978340208530426, 0.307151734828949, 0.36142686009407043, 0.26026037335395813], [0.002192062558606267, 0.206104576587677, 0.4561813175678253, 0.0025961906649172306], [0.37231361865997314, 0.3578506410121918, 0.4104750454425812, 0.19113443791866302], [0.47357606887817383, 0.14654675126075745, 0.38249528408050537, 0.2801401913166046], [0.33489128947257996, 0.27414944767951965, 0.386406809091568, 0.349480539560318]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([6], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ad55397e73d632759a1419b176ad70a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93eb03926e80aaf89790fd0d70d3f4b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b8d5dc264570c8b67df30d8045d71c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e130c493186dbf2a2f48cd0515ede4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ffaa199627160097224077a4a8627b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e04cea6c0fbaa9c78ef4f5bbbbb37c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()