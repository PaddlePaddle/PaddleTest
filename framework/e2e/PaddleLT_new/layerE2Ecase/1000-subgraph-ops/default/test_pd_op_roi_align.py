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


class TestPrimitiveOp_888be088049a240b1019fac164f97089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5da554bc3fca8434943d17c9652671c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.17662222683429718, 0.4799039959907532, 0.3506110608577728, 0.2692452371120453], [0.30085206031799316, 0.36706987023353577, 0.08642890304327011, 0.01676788739860058]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_abe7ac111685f225cc303439d6aa71e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.018669940531253815, 0.06625571846961975, 0.3648173213005066, 0.014738636091351509], [0.19450293481349945, 0.35408738255500793, 0.127450093626976, 0.4606354832649231]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_f37172b73d2d74de0f81abc7869ae2a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10d7f9cdbfcce2d8b3982046448f10ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.4446417987346649, 0.3217782974243164, 0.22582820057868958, 0.4738568663597107], [0.18493904173374176, 0.08554968237876892, 0.04868201166391373, 0.13738179206848145]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_9c8e19ea5866fe1c74633b633348f4f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.43867161870002747, 0.34236693382263184, 0.08963692933320999, 0.27902036905288696], [0.3868449032306671, 0.3657459020614624, 0.4508436322212219, 0.11935139447450638], [0.013107744045555592, 0.2996593713760376, 0.0653977021574974, 0.3356562554836273], [0.4208671748638153, 0.2477605640888214, 0.047215186059474945, 0.08065006136894226], [0.2426077425479889, 0.4226152300834656, 0.0017632709350436926, 0.412915974855423], [0.148656964302063, 0.489155650138855, 0.44146063923835754, 0.48710185289382935], [0.446685791015625, 0.07128840684890747, 0.38714897632598877, 0.006745404563844204]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_bb828de13d9376bb9c38fb4a6b2f3bcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.11424048244953156, 0.3029647469520569, 0.13629528880119324, 0.46353811025619507], [0.40731096267700195, 0.15129809081554413, 0.04926984757184982, 0.3863285779953003], [0.36206063628196716, 0.17123213410377502, 0.2247663289308548, 0.0035095566418021917], [0.4017271399497986, 0.4620095193386078, 0.1621353179216385, 0.10692046582698822], [0.23194661736488342, 0.273383229970932, 0.19963407516479492, 0.03132865950465202], [0.09034047275781631, 0.1794315129518509, 0.07394567877054214, 0.11570005118846893]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_b89152024e362346f41d4d3cd360b9d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.4747256636619568, 0.19396467506885529, 0.006264861673116684, 0.36866122484207153], [0.2978251874446869, 0.3300935626029968, 0.19536623358726501, 0.06849707663059235], [0.4240475296974182, 0.424383282661438, 0.3205764591693878, 0.07464299350976944]], dtype='float32').reshape([3, 4]),
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


class TestPrimitiveOp_1c54d9a54acda45c059e70f7a6ac88f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10d7f9cdbfcce2d8b3982046448f10ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.14199818670749664, 0.04861406236886978, 0.38133692741394043, 0.05661538988351822], [0.0675225555896759, 0.1788635551929474, 0.46543440222740173, 0.41755279898643494]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_d8ea5cad88080a976562ced70237073a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.3007311224937439, 0.07645222544670105, 0.49056655168533325, 0.10885652154684067]], dtype='float32').reshape([1, 4]),
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


class TestPrimitiveOp_b10739af110abf47fe0bcc12c4071d68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.2660581171512604, 0.007315840106457472, 0.22027195990085602, 0.018314115703105927], [0.14150077104568481, 0.4878149926662445, 0.003385629504919052, 0.4338425099849701], [0.4937858283519745, 0.31834039092063904, 0.2641705274581909, 0.17065684497356415], [0.23337814211845398, 0.26186394691467285, 0.30601999163627625, 0.2453814297914505], [0.0447695218026638, 0.00456420099362731, 0.1720564067363739, 0.4235546886920929], [0.4087570309638977, 0.10791429877281189, 0.3522658348083496, 0.14169463515281677], [0.08416858315467834, 0.3609386086463928, 0.43241164088249207, 0.37270158529281616]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_ddc136f1f6a817728136acf8ac1738c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.4652860760688782, 0.039080239832401276, 0.42288684844970703, 0.0058063846081495285]], dtype='float32').reshape([1, 4]),
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


class TestPrimitiveOp_532e63906b9a55ae059433ae8ca28368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.1648794263601303, 0.469783753156662, 0.24081149697303772, 0.2175903618335724], [0.39655786752700806, 0.33049485087394714, 0.2591630816459656, 0.34178102016448975], [0.1480017751455307, 0.02110651321709156, 0.48171260952949524, 0.46052899956703186], [0.439046710729599, 0.1369829773902893, 0.189821258187294, 0.2592080235481262], [0.008663302287459373, 0.23973485827445984, 0.3611301779747009, 0.014702463522553444]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_334d1f02799b8ee17f084487b972a34c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.3982559144496918, 0.014606457203626633, 0.14966696500778198, 0.015552563592791557], [0.03265177458524704, 0.3376113474369049, 0.2607705295085907, 0.43766531348228455], [0.23449233174324036, 0.02971261739730835, 0.36613699793815613, 0.36435553431510925], [0.035463057458400726, 0.19821660220623016, 0.10969212651252747, 0.16864947974681854], [0.1744171679019928, 0.14180982112884521, 0.09694315493106842, 0.3655495047569275], [0.2059173434972763, 0.31622010469436646, 0.27419108152389526, 0.07211241126060486], [0.46466243267059326, 0.29169726371765137, 0.18111707270145416, 0.49232813715934753]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_94b0c066205462ba767e01913fb614cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.4769362211227417, 0.2369346171617508, 0.46974578499794006, 0.4170452356338501], [0.4785066545009613, 0.060912519693374634, 0.21230709552764893, 0.01750330440700054], [0.11079324036836624, 0.3878647983074188, 0.2297121286392212, 0.07275450229644775], [0.3869275748729706, 0.4108844995498657, 0.463040828704834, 0.40595579147338867], [0.32531681656837463, 0.30542126297950745, 0.10497302561998367, 0.23128855228424072], [0.02544524520635605, 0.08206713199615479, 0.4639264941215515, 0.03456242009997368], [0.16951483488082886, 0.074600949883461, 0.2729337513446808, 0.39293229579925537]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_9b366501d0daf863d46b68972f6ceccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.28772470355033875, 0.13713136315345764, 0.16108132898807526, 0.10373386740684509]], dtype='float32').reshape([1, 4]),
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


class TestPrimitiveOp_d98c7936618ae4c6bda9aa9447539038(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.10247860848903656, 0.09309497475624084, 0.1762848198413849, 0.4288789927959442], [0.09719467163085938, 0.2945011258125305, 0.2505485713481903, 0.04018883407115936], [0.0990212932229042, 0.3297467827796936, 0.3885916769504547, 0.3770819306373596], [0.49362167716026306, 0.10047370940446854, 0.4849914014339447, 0.474293053150177], [0.45900994539260864, 0.0008709044777788222, 0.2990599572658539, 0.3226492404937744]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_121db04a9dfdab1973f8d2e4c675e7de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.45812860131263733, 0.34698742628097534, 0.250236451625824, 0.221030592918396]], dtype='float32').reshape([1, 4]),
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


class TestPrimitiveOp_0b038ee4f796cf56bb91431e2ce45b50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.3117319345474243, 0.4302293360233307, 0.33149445056915283, 0.42980971932411194]], dtype='float32').reshape([1, 4]),
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


class TestPrimitiveOp_b9ec947d44d950a8ce54259e245ad238(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.06020370498299599, 0.19599413871765137, 0.015415714122354984, 0.4263978600502014]], dtype='float32').reshape([1, 4]),
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


class TestPrimitiveOp_8be9e08e6b438fe9a4e46655f73eae49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a5ebde34d215d91a82e48405d13a5ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.05851532891392708, 0.05855214595794678, 0.0004865849041379988, 0.074506975710392], [0.2852267622947693, 0.24892088770866394, 0.24263961613178253, 0.04706287384033203], [0.3192131519317627, 0.1558162122964859, 0.34629324078559875, 0.09156456589698792], [0.021957047283649445, 0.17742015421390533, 0.29279401898384094, 0.33543986082077026], [0.10557131469249725, 0.4661080539226532, 0.35490748286247253, 0.23102009296417236], [0.3108401894569397, 0.08042491972446442, 0.35233521461486816, 0.48459362983703613], [0.33629730343818665, 0.31278395652770996, 0.09472471475601196, 0.08141104131937027]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_ffdcd1ecd230db2b7692a9d9de44cfb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596ac55d96e3b18170342caa7054fdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.35814958810806274, 0.20164504647254944, 0.058809105306863785, 0.21588782966136932], [0.09151577949523926, 0.4182908833026886, 0.2823256254196167, 0.3646542429924011], [0.14444118738174438, 0.13877679407596588, 0.19844818115234375, 0.0030155074782669544], [0.2838691473007202, 0.03378880396485329, 0.4606785774230957, 0.3735581040382385], [0.1633516550064087, 0.22554175555706024, 0.09943287074565887, 0.34326329827308655], [0.32687854766845703, 0.4422873258590698, 0.02415948174893856, 0.14546100795269012]], dtype='float32').reshape([6, 4]),
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