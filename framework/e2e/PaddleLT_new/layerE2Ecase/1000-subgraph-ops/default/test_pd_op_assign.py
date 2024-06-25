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



class PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e710312569a9ae11fdca3a6240500d3a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4fb791cbcd3fd072efde52ba99b450a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.5973773002624512]], [[0.6169924736022949]], [[0.8154162764549255]], [[0.6751572489738464]], [[0.7172725200653076]], [[0.797681450843811]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_0dccf564d6ac1e349a95d531efb3a0dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.5493918657302856]], [[0.8088148236274719]], [[0.524810254573822]], [[0.6945334672927856]], [[0.6179171800613403]], [[0.6792595386505127]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_0344873045403973f5561db6ba9f7074(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7b76d0023756cb43fa432ace51cd2ffe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 720, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4d30ce7c34dfa46c882326fdb111040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b76d0023756cb43fa432ace51cd2ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d30ce7c34dfa46c882326fdb111040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b76d0023756cb43fa432ace51cd2ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d30ce7c34dfa46c882326fdb111040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b76d0023756cb43fa432ace51cd2ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d30ce7c34dfa46c882326fdb111040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b76d0023756cb43fa432ace51cd2ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5383b8b4b670f8f98a80039f84e5d7c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_469646532d81dbe66d6c4045c7623276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5383b8b4b670f8f98a80039f84e5d7c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_469646532d81dbe66d6c4045c7623276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5383b8b4b670f8f98a80039f84e5d7c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_469646532d81dbe66d6c4045c7623276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5383b8b4b670f8f98a80039f84e5d7c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_469646532d81dbe66d6c4045c7623276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5383b8b4b670f8f98a80039f84e5d7c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f067d13d8993b7e4d5b422deaeae731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_ae0bcf0901cb7bd9dacc7270466b6e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(4096, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6e8f41e1a2deebbec322f41879ffaa71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(16384, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a3abddea29f7af263cde72f085aa2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_536db7d4dfd1044b0e3221aa4beb450a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7999748bb241d7cf17dd9b140da5e1d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_c9712aa900ff7ceb5d6f5e8fcbfe757a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[12096, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc77138eec8bb05a925b46680d99c7cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9712aa900ff7ceb5d6f5e8fcbfe757a
    def get_inputs(self):
        return [
            paddle.uniform([12096, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_62f2d8f44af97838f003ae71b035b470(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27a799c4441fe9b2fd361b7e0e95a55c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62f2d8f44af97838f003ae71b035b470
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27a799c4441fe9b2fd361b7e0e95a55c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62f2d8f44af97838f003ae71b035b470
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27a799c4441fe9b2fd361b7e0e95a55c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62f2d8f44af97838f003ae71b035b470
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d2bcfae0d7ac15ec3263e2c7bfa2aeb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1cf7fd513881a1521a3130ce76aacd8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1f8cad836271d70676d8764c0f390e2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_743f6b1d50b7849cefcc498c481767de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f8cad836271d70676d8764c0f390e2c
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_05c7035ba40d0076b1c9f5aecd319e34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2db133f166f6f8d3cbcac585e4b7ed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c7035ba40d0076b1c9f5aecd319e34
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2db133f166f6f8d3cbcac585e4b7ed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c7035ba40d0076b1c9f5aecd319e34
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4196e0c456477bf64ee52f42c1bee59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bbb4921bbee3b56b91dfa3f9ac031979(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 300, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3001dcfebca980fcdb01576e46ca35f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbb4921bbee3b56b91dfa3f9ac031979
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_9cbf088de75b4e114159f431b225e2b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fc2b5b453663ec1780c641e50f2f509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9cbf088de75b4e114159f431b225e2b5
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4fc2b5b453663ec1780c641e50f2f509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9cbf088de75b4e114159f431b225e2b5
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b4196e0c456477bf64ee52f42c1bee59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_9cab7bbea5426562934adb5a24273896(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5376, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eacb876c80f02cd3d74fc02dab60d5de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9cab7bbea5426562934adb5a24273896
    def get_inputs(self):
        return [
            paddle.uniform([5376, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_34f3e061dc2d32ecef2a19f6a1d7bfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.5539252161979675]], [[0.6346803307533264]], [[0.7276504635810852]], [[0.5098108649253845]], [[0.7054385542869568]], [[0.6356890797615051]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_4b1fdcab8d0058b4cde1892a6791448c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.5134385228157043]], [[0.6661975979804993]], [[0.6664236187934875]], [[0.6304938793182373]], [[0.7162249088287354]], [[0.5771429538726807]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ae3754e4137218b38df5c51b1fc01122(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a3abddea29f7af263cde72f085aa2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6a3abddea29f7af263cde72f085aa2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1cf7fd513881a1521a3130ce76aacd8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_e7216a894ea006f29eaed2aa80c1ee11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34b976e97f18f92c1256f92f3e9a64d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7216a894ea006f29eaed2aa80c1ee11
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34b976e97f18f92c1256f92f3e9a64d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7216a894ea006f29eaed2aa80c1ee11
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_27a799c4441fe9b2fd361b7e0e95a55c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62f2d8f44af97838f003ae71b035b470
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27a799c4441fe9b2fd361b7e0e95a55c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62f2d8f44af97838f003ae71b035b470
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_096eed67deda7dbec795f83a7d426503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_096eed67deda7dbec795f83a7d426503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_794149809b9fdc04ce51fa5624ea9b64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4885d75e1673b46c9f3de93349bcd0dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_794149809b9fdc04ce51fa5624ea9b64
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4885d75e1673b46c9f3de93349bcd0dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_794149809b9fdc04ce51fa5624ea9b64
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b7d9d2e3e1e4aadb77c080cdc710fa57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_794149809b9fdc04ce51fa5624ea9b64
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7d9d2e3e1e4aadb77c080cdc710fa57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_794149809b9fdc04ce51fa5624ea9b64
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b4196e0c456477bf64ee52f42c1bee59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_096eed67deda7dbec795f83a7d426503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_096eed67deda7dbec795f83a7d426503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d16762ebd1595963e380a577bd551f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d16762ebd1595963e380a577bd551f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d16762ebd1595963e380a577bd551f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d16762ebd1595963e380a577bd551f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d16762ebd1595963e380a577bd551f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d16762ebd1595963e380a577bd551f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d16762ebd1595963e380a577bd551f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_20f037ffabddcc664e0cd7500716732e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_342def8e9bd03ff2a2ea83b323a332a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.03985670953989029, 0.3563137352466583, 0.0256974995136261, -0.07077597081661224, 0.2433735728263855, 0.24213233590126038], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d383ffe5eefc04cfc890fea714038ec0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10679596662521362, -0.2945631742477417, 0.04922857880592346, -0.2021600604057312, -0.10806180536746979, -0.09570963680744171], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_caaea5d2fa919f2fb29ffcae3712e463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32777875661849976, 0.19396711885929108, 0.03677868843078613, 0.2845810651779175, 0.025402367115020752, 0.22408275306224823], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_bbf1b1b1c4bf640f9c8e5f2d80d61144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19039610028266907, 0.037546172738075256, -0.22206011414527893, -0.21200229227542877, 0.027823418378829956, 0.06056720018386841], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_39c15a3744014677a6f1f372fb712340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3119310736656189, 0.38255348801612854, 0.1359562873840332, 0.4140319228172302, 0.2433735728263855, 0.3911452889442444], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_44be9af8c446ce7af63bc9536f17bd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3801032304763794, 0.0, 0.1558144986629486, 0.0, 0.10956054925918579, 0.06975878775119781], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_46447d8b484d38d90f019647b15ae079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a3abddea29f7af263cde72f085aa2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_536db7d4dfd1044b0e3221aa4beb450a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7999748bb241d7cf17dd9b140da5e1d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_063d404df1bc2153dbf3d1cd4e63e801(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8400, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10daeb1e43c3da9715fd7a805801bc7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_063d404df1bc2153dbf3d1cd4e63e801
    def get_inputs(self):
        return [
            paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_f3e973b8f62d9a928a3d56be83e98604(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1de55ef8b4753e71cbe65ea042ff1324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3e973b8f62d9a928a3d56be83e98604
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1de55ef8b4753e71cbe65ea042ff1324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3e973b8f62d9a928a3d56be83e98604
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1de55ef8b4753e71cbe65ea042ff1324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3e973b8f62d9a928a3d56be83e98604
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b4669f32cf9267a017c62e5a9674eeff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b4669f32cf9267a017c62e5a9674eeff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b4669f32cf9267a017c62e5a9674eeff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b4669f32cf9267a017c62e5a9674eeff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b4669f32cf9267a017c62e5a9674eeff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b4669f32cf9267a017c62e5a9674eeff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b4669f32cf9267a017c62e5a9674eeff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5f59e52056d56dd35a3bc23295879d03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5f59e52056d56dd35a3bc23295879d03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5f59e52056d56dd35a3bc23295879d03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1cf7fd513881a1521a3130ce76aacd8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2f067d13d8993b7e4d5b422deaeae731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_ae0bcf0901cb7bd9dacc7270466b6e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(4096, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6e8f41e1a2deebbec322f41879ffaa71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(16384, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1cf7fd513881a1521a3130ce76aacd8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_270d60ebb46676dd4ce60115d0031c76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d6ed67ef9e2f71981daf365d3cfcd46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_270d60ebb46676dd4ce60115d0031c76
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d6ed67ef9e2f71981daf365d3cfcd46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_270d60ebb46676dd4ce60115d0031c76
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d6ed67ef9e2f71981daf365d3cfcd46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_270d60ebb46676dd4ce60115d0031c76
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b4196e0c456477bf64ee52f42c1bee59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_9e6877079f8d1d16e4a3d5fbbd5e3ab1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d6ed5c2c63c757192f2f0a8da6ff285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e6877079f8d1d16e4a3d5fbbd5e3ab1
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d6ed5c2c63c757192f2f0a8da6ff285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e6877079f8d1d16e4a3d5fbbd5e3ab1
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d6ed5c2c63c757192f2f0a8da6ff285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e6877079f8d1d16e4a3d5fbbd5e3ab1
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d6ed5c2c63c757192f2f0a8da6ff285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e6877079f8d1d16e4a3d5fbbd5e3ab1
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_096eed67deda7dbec795f83a7d426503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_096eed67deda7dbec795f83a7d426503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b4196e0c456477bf64ee52f42c1bee59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_10ac385ce3f863952d411d3716f1d3ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39c67eea9ef5fd090ed6be2238483b0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10ac385ce3f863952d411d3716f1d3ab
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39c67eea9ef5fd090ed6be2238483b0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10ac385ce3f863952d411d3716f1d3ab
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39c67eea9ef5fd090ed6be2238483b0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10ac385ce3f863952d411d3716f1d3ab
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39c67eea9ef5fd090ed6be2238483b0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10ac385ce3f863952d411d3716f1d3ab
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf6e704687a7f86ba7a9439819bcde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f91e5f693325a5d5cd24ccb2a69d0a36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_d7feab9632b578bb81b0b4dc7f519d74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96429e35351502eefc05831dbfdec92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7feab9632b578bb81b0b4dc7f519d74
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96429e35351502eefc05831dbfdec92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7feab9632b578bb81b0b4dc7f519d74
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96429e35351502eefc05831dbfdec92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7feab9632b578bb81b0b4dc7f519d74
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96429e35351502eefc05831dbfdec92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7feab9632b578bb81b0b4dc7f519d74
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96429e35351502eefc05831dbfdec92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7feab9632b578bb81b0b4dc7f519d74
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_3d2683a2ac9b5c52e553731fbb2842eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 600, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b13fa2affba9f08bcc94473a925586b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d2683a2ac9b5c52e553731fbb2842eb
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b13fa2affba9f08bcc94473a925586b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d2683a2ac9b5c52e553731fbb2842eb
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d30ce7c34dfa46c882326fdb111040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b76d0023756cb43fa432ace51cd2ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d30ce7c34dfa46c882326fdb111040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b76d0023756cb43fa432ace51cd2ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d30ce7c34dfa46c882326fdb111040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b76d0023756cb43fa432ace51cd2ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d30ce7c34dfa46c882326fdb111040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b76d0023756cb43fa432ace51cd2ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_469646532d81dbe66d6c4045c7623276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5383b8b4b670f8f98a80039f84e5d7c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_469646532d81dbe66d6c4045c7623276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5383b8b4b670f8f98a80039f84e5d7c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_469646532d81dbe66d6c4045c7623276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5383b8b4b670f8f98a80039f84e5d7c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_469646532d81dbe66d6c4045c7623276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5383b8b4b670f8f98a80039f84e5d7c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_c15f3c6a926631589f4af56d044b56d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4a6606e4ab0b467431f17fb827022ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_41a3f9ce11722552230610af3ae91ec5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17a0cf236aecb9fd1e796f445b2d913c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_4a9d52793c60af0a77e61aac4e69d110(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6069, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31599f7d583ed0536f9bece739f0621e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a9d52793c60af0a77e61aac4e69d110
    def get_inputs(self):
        return [
            paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7eba6d4fdeb0894df42fc55fcebf78e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7eba6d4fdeb0894df42fc55fcebf78e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7eba6d4fdeb0894df42fc55fcebf78e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0cab323ebe4afc5c43f775b40e543000(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d69abd510b726d9baee1beea6d29096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cab323ebe4afc5c43f775b40e543000
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0780007466673851]], [[0.21969814598560333]], [[0.1599072813987732]], [[0.10114734619855881]], [[0.029051214456558228]], [[0.13129179179668427]], [[0.41286569833755493]], [[0.16294817626476288]], [[0.14811532199382782]], [[0.06489606201648712]], [[0.327319473028183]], [[0.24952372908592224]], [[0.2902521789073944]], [[0.11505094915628433]], [[0.3753133714199066]], [[0.21967285871505737]], [[0.4150799512863159]], [[0.09354554116725922]], [[0.08100428432226181]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d4a6606e4ab0b467431f17fb827022ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c15f3c6a926631589f4af56d044b56d9
    def get_inputs(self):
        return [
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9a57db4c10664c642205a4c0cfdcaf43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6e8f41e1a2deebbec322f41879ffaa71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(16384, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_ae0bcf0901cb7bd9dacc7270466b6e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(4096, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2f067d13d8993b7e4d5b422deaeae731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_58bf3f0e8135e26c666586df3dc68252(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_12eb573582348e38de166f3f5afee02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_98a1cf318200f678f4e7fe8e889a69e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8dbe5d42fcc1779bb39e98347154ff7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a1cf318200f678f4e7fe8e889a69e8
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dbe5d42fcc1779bb39e98347154ff7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a1cf318200f678f4e7fe8e889a69e8
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7eba6d4fdeb0894df42fc55fcebf78e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7eba6d4fdeb0894df42fc55fcebf78e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7eba6d4fdeb0894df42fc55fcebf78e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_91ae7c636312797ca25388c42fe722c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ec04044d71c97f6af9fda990cf13eec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91ae7c636312797ca25388c42fe722c0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.21206022799015045]], [[0.22143475711345673]], [[0.07676990330219269]], [[0.29928699135780334]], [[0.16691668331623077]], [[0.16907809674739838]], [[0.46916866302490234]], [[0.16731517016887665]], [[0.15742270648479462]], [[0.42591360211372375]], [[0.12636816501617432]], [[0.2972332835197449]], [[0.4920099973678589]], [[0.21693216264247894]], [[0.15520325303077698]], [[0.03897607699036598]], [[0.29736077785491943]], [[0.2954310476779938]], [[0.2732740640640259]], [[0.461588054895401]], [[0.29754093289375305]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2be7b0b01fe5cc6ec4f92b72a20e5508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_039165d350c48dc18270375fe4de6c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4196e0c456477bf64ee52f42c1bee59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7ac4917226b8cf4ae02d33202f04af30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_3ff313bc7e9a3cc6c568d8563e121a38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36d541c31b886ac8e248b7cf9dc11134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ff313bc7e9a3cc6c568d8563e121a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4196e0c456477bf64ee52f42c1bee59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_07c4d0d7c82ba6424900541ad5280edf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe6f09c612e8fb778981d8f1f268f9ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c4d0d7c82ba6424900541ad5280edf
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe6f09c612e8fb778981d8f1f268f9ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c4d0d7c82ba6424900541ad5280edf
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1cf7fd513881a1521a3130ce76aacd8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_59b4dfc57f911a4f3d703749ee5ca016(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4011bd4f510f01afa1a9ef70a253861c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59b4dfc57f911a4f3d703749ee5ca016
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4011bd4f510f01afa1a9ef70a253861c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59b4dfc57f911a4f3d703749ee5ca016
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2f067d13d8993b7e4d5b422deaeae731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_ae0bcf0901cb7bd9dacc7270466b6e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(4096, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6e8f41e1a2deebbec322f41879ffaa71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(16384, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4196e0c456477bf64ee52f42c1bee59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb42cbb88f82d7fd643d297633ff27d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0344873045403973f5561db6ba9f7074
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d30ce7c34dfa46c882326fdb111040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b76d0023756cb43fa432ace51cd2ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d30ce7c34dfa46c882326fdb111040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b76d0023756cb43fa432ace51cd2ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d30ce7c34dfa46c882326fdb111040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b76d0023756cb43fa432ace51cd2ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d30ce7c34dfa46c882326fdb111040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b76d0023756cb43fa432ace51cd2ffe
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_469646532d81dbe66d6c4045c7623276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5383b8b4b670f8f98a80039f84e5d7c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_469646532d81dbe66d6c4045c7623276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5383b8b4b670f8f98a80039f84e5d7c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_469646532d81dbe66d6c4045c7623276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5383b8b4b670f8f98a80039f84e5d7c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_469646532d81dbe66d6c4045c7623276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5383b8b4b670f8f98a80039f84e5d7c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_9c35af257fb04870d5b2e06d9659123a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_764a8ec62b7781ff172c6d5840c649bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c35af257fb04870d5b2e06d9659123a
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_764a8ec62b7781ff172c6d5840c649bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c35af257fb04870d5b2e06d9659123a
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_6c0829870f2e761b16a514e749f5b298(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 180, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76d84c316f9c0fcf87055d34a8e8271d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c0829870f2e761b16a514e749f5b298
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76d84c316f9c0fcf87055d34a8e8271d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c0829870f2e761b16a514e749f5b298
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_096eed67deda7dbec795f83a7d426503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_096eed67deda7dbec795f83a7d426503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_099b3faea59495106fb26a6e9b1b7683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ff313bc7e9a3cc6c568d8563e121a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_372a3ae6420a8b0fc2b85c3490a8b477(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_9f10eb8fd94a8f483c2046d591c0dae8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6804, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e4940d05114cf0841c8a389876aac26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f10eb8fd94a8f483c2046d591c0dae8
    def get_inputs(self):
        return [
            paddle.uniform([6804, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_f170b21a6c7f7d041b9ad42afb19b1d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61e449c20d5e73c26a3c60d8237d52fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f170b21a6c7f7d041b9ad42afb19b1d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_61e449c20d5e73c26a3c60d8237d52fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f170b21a6c7f7d041b9ad42afb19b1d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b4196e0c456477bf64ee52f42c1bee59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7da30a7fdd54427a207140db93b35a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd49584dc26ee2827089882e002242b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da8a550fe7d711ba68d2647b28284ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b4196e0c456477bf64ee52f42c1bee59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_aa845eb578bf2465c7f6bf89192d636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135b0daa50de5a83c73fbe67326b63a5
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9a0dbc4ce4df03dda29467bc9cdd02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f067d13d8993b7e4d5b422deaeae731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_ae0bcf0901cb7bd9dacc7270466b6e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(4096, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6e8f41e1a2deebbec322f41879ffaa71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97262623fe8fcc128e38ba0fb1de6cf2
    def get_inputs(self):
        return [
            paddle.to_tensor(16384, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ff8a45ed268edb287b469683eb07514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5b4c9e2318cb7a069c31b14a4f7fd29
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04702d2f1dba47d4c5002b7a868232a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9be8d314257821a8803c1f80c50cb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04afeec50e5ca86d41d8ee9d9762448f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59128d6e73ade9d91eace59d06daed4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_47239135755ad63874977f9d5d56ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_594fed38237c4d4e3c2c4ca2f77f8aa1
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()