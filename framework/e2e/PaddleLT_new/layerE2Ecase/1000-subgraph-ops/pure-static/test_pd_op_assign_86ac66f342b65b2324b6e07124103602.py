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


class PrimitiveOp_a6685ae1d33e222cb00988a16f9a9474(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93f78b9fa12e423720707a7dbf710dde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6685ae1d33e222cb00988a16f9a9474
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.70428466796875]], [[0.5585848689079285]], [[0.5928927659988403]], [[0.7616262435913086]], [[0.6079586148262024]], [[0.515092670917511]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_4ebb58b74569c69fb69a5b7db1f9000d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6685ae1d33e222cb00988a16f9a9474
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.6252842545509338]], [[0.7332195043563843]], [[0.7346566915512085]], [[0.7151126861572266]], [[0.8184952139854431]], [[0.7468409538269043]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
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


class PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class PrimitiveOp_d52d496506f4698dc2eed2de9a84ab91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 48, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbc330fb4bf8ee0492f9456f6079c9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d52d496506f4698dc2eed2de9a84ab91
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbc330fb4bf8ee0492f9456f6079c9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d52d496506f4698dc2eed2de9a84ab91
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbc330fb4bf8ee0492f9456f6079c9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d52d496506f4698dc2eed2de9a84ab91
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


class PrimitiveOp_3909488ee7020f28cbfa053ae7720a57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 40, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a4fb1c663677150e78a714023897a2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3909488ee7020f28cbfa053ae7720a57
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a4fb1c663677150e78a714023897a2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3909488ee7020f28cbfa053ae7720a57
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class TestPrimitiveOp_29caac3977067601aa9f4d409b8203b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6685ae1d33e222cb00988a16f9a9474
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.7049703598022461]], [[0.8226004838943481]], [[0.5391098856925964]], [[0.5475325584411621]], [[0.5748777985572815]], [[0.8003376126289368]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_e408d2e2cd7180ce3da30baee67077be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6685ae1d33e222cb00988a16f9a9474
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.5529621243476868]], [[0.6618142127990723]], [[0.6048946976661682]], [[0.7230639457702637]], [[0.5504274368286133]], [[0.5951051115989685]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class PrimitiveOp_0b473fdec2df73c0d18e8513de407fb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80e9b9034533730c60bfc0dee10eedfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b473fdec2df73c0d18e8513de407fb1
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80e9b9034533730c60bfc0dee10eedfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b473fdec2df73c0d18e8513de407fb1
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


class TestPrimitiveOp_bbc330fb4bf8ee0492f9456f6079c9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d52d496506f4698dc2eed2de9a84ab91
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbc330fb4bf8ee0492f9456f6079c9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d52d496506f4698dc2eed2de9a84ab91
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


class PrimitiveOp_3b354d84a1e81d9d0ac39ba5e9d2c7b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 120, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a12aa5707ae2dd5b8d18e91f7e61bf95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b354d84a1e81d9d0ac39ba5e9d2c7b6
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a12aa5707ae2dd5b8d18e91f7e61bf95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b354d84a1e81d9d0ac39ba5e9d2c7b6
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


class PrimitiveOp_e54495092217f3c3859eea6c338a6140(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 120, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1dd00a451a2aae5fa5e31b20e1a67296(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e54495092217f3c3859eea6c338a6140
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dd00a451a2aae5fa5e31b20e1a67296(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e54495092217f3c3859eea6c338a6140
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class PrimitiveOp_74fd7ea5b111181064479d36b18c0d4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0d3573c8e47ef0854b99f2eda1051fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74fd7ea5b111181064479d36b18c0d4c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.17463308572769165, -0.2213263213634491, -0.3390448987483978, -0.017620444297790527, 0.368111252784729, 0.18772169947624207], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_f60955aa6243e2ac2fc263f4c3fee0f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74fd7ea5b111181064479d36b18c0d4c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.006831318140029907, -0.05255473405122757, 0.2690444588661194, 0.09878036379814148, 0.10982172191143036, -0.16934990882873535], dtype='float32').reshape([6]),
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


class TestPrimitiveOp_7c4d188a04a77fd16ee8055e70dee6a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74fd7ea5b111181064479d36b18c0d4c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10851454734802246, 0.08410894870758057, -0.13825735449790955, -0.1745946705341339, 0.07062837481498718, 0.14177775382995605], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_8ad94896659f80d02c234cff12f4ed82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74fd7ea5b111181064479d36b18c0d4c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.2124403417110443, 0.23160921037197113, 0.29485806822776794, -0.2684791386127472, 0.286822646856308, 0.29468977451324463], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_28bcff90685bdd68dcc62109871c41a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74fd7ea5b111181064479d36b18c0d4c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3788352608680725, 0.11308163404464722, 0.2824884057044983, 0.26669132709503174, 0.368111252784729, 0.2842792570590973], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ecad90324fb8d664b3991e8de87cb720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74fd7ea5b111181064479d36b18c0d4c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22137197852134705, 0.3965763449668884, 0.45734041929244995, 0.25495967268943787, 0.42001980543136597, 0.262728750705719], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_fda8a4fc72aee88a9287c43439106dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcd2e12a90121c5dde14689334bd60
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_abe357e6396399cec9b091bf8a9139c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74fd7ea5b111181064479d36b18c0d4c
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class PrimitiveOp_462f75bd6e75d2467986885e19634a43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 160, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_228b419f241ec9d54d71d86fb3966cb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_462f75bd6e75d2467986885e19634a43
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_228b419f241ec9d54d71d86fb3966cb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_462f75bd6e75d2467986885e19634a43
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_228b419f241ec9d54d71d86fb3966cb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_462f75bd6e75d2467986885e19634a43
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class PrimitiveOp_a6149fde48fb734da8d75ac376a70bda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 80, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4716644f618c97379f35104c76f14771(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6149fde48fb734da8d75ac376a70bda
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4716644f618c97379f35104c76f14771(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6149fde48fb734da8d75ac376a70bda
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4716644f618c97379f35104c76f14771(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6149fde48fb734da8d75ac376a70bda
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class PrimitiveOp_69c58b4bf29b5d920549eed780ce8c74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 300, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8f81970ab7429b54d064b41921ca2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69c58b4bf29b5d920549eed780ce8c74
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8f81970ab7429b54d064b41921ca2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69c58b4bf29b5d920549eed780ce8c74
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8f81970ab7429b54d064b41921ca2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69c58b4bf29b5d920549eed780ce8c74
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8f81970ab7429b54d064b41921ca2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69c58b4bf29b5d920549eed780ce8c74
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class PrimitiveOp_1a024146634a8a071d526a4e2fa51960(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 90, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a378797115e1c57ff54a2419be74204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a024146634a8a071d526a4e2fa51960
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a378797115e1c57ff54a2419be74204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a024146634a8a071d526a4e2fa51960
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a378797115e1c57ff54a2419be74204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a024146634a8a071d526a4e2fa51960
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a378797115e1c57ff54a2419be74204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a024146634a8a071d526a4e2fa51960
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class PrimitiveOp_06382195b1a437469f4ce8f5a4c9a0d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 144, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66253f1f0bb6eaca7aef3363e446ee40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06382195b1a437469f4ce8f5a4c9a0d0
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66253f1f0bb6eaca7aef3363e446ee40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06382195b1a437469f4ce8f5a4c9a0d0
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66253f1f0bb6eaca7aef3363e446ee40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06382195b1a437469f4ce8f5a4c9a0d0
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66253f1f0bb6eaca7aef3363e446ee40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06382195b1a437469f4ce8f5a4c9a0d0
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66253f1f0bb6eaca7aef3363e446ee40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06382195b1a437469f4ce8f5a4c9a0d0
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class PrimitiveOp_748d47965c098cf5ac498ac758431b72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 600, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c19e6b9c73ae6fbf3217d3e1bb73e20e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_748d47965c098cf5ac498ac758431b72
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c19e6b9c73ae6fbf3217d3e1bb73e20e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_748d47965c098cf5ac498ac758431b72
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class PrimitiveOp_5d20672bd73ab784e437a91e81bda090(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77939b9bbf2b261e3c67fb3912ea0c95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d20672bd73ab784e437a91e81bda090
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


class TestPrimitiveOp_532f5a5f12c508c21029843f1d734556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cab323ebe4afc5c43f775b40e543000
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.34988516569137573]], [[0.004709410481154919]], [[0.037150993943214417]], [[0.10848682373762131]], [[0.22562730312347412]], [[0.14453700184822083]], [[0.3420802354812622]], [[0.22003598511219025]], [[0.3644031882286072]], [[0.3968456983566284]], [[0.057300906628370285]], [[0.444746196269989]], [[0.07461818307638168]], [[0.3230685889720917]], [[0.16343358159065247]], [[0.3999313414096832]], [[0.22733542323112488]], [[0.31414327025413513]], [[0.05352018028497696]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_6692e6e4e727053cc5983e82505894e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ab8f94dda50fcb3c58a3d95a9c2972
    def get_inputs(self):
        return [
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


class PrimitiveOp_b63e69ef803819c73406af1183dd8013(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_946f91c1a94c3b0d48f21dffe6708549(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b63e69ef803819c73406af1183dd8013
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


class PrimitiveOp_606680daea9eb2f05cc016f14b5f598b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 36, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ccc75c2d1dd931c8d8eef201151a566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_606680daea9eb2f05cc016f14b5f598b
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ccc75c2d1dd931c8d8eef201151a566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_606680daea9eb2f05cc016f14b5f598b
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


class TestPrimitiveOp_3259c27ba6171704c6f4b4282c5e4c5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91ae7c636312797ca25388c42fe722c0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.23186716437339783]], [[0.4627135097980499]], [[0.44898757338523865]], [[0.16095741093158722]], [[0.21885348856449127]], [[0.047801945358514786]], [[0.3218590021133423]], [[0.1415168195962906]], [[0.16289526224136353]], [[0.010165276937186718]], [[0.3935929834842682]], [[0.4789142906665802]], [[0.40032556653022766]], [[0.19515080749988556]], [[0.19329068064689636]], [[0.24012264609336853]], [[0.3002344071865082]], [[0.3082791566848755]], [[0.4747837483882904]], [[0.24912673234939575]], [[0.40240731835365295]]]], dtype='float32').reshape([1, 21, 1, 1]),
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class PrimitiveOp_5fc6e3ed7d5dc19596f61d6a94097d61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24f31c54fab327de71f15b93be409084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fc6e3ed7d5dc19596f61d6a94097d61
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


class PrimitiveOp_ebeb817b7255d2868459f289ef37e563(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 60, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_484dbf0bcf338c944974b0011e4b1837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebeb817b7255d2868459f289ef37e563
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_484dbf0bcf338c944974b0011e4b1837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebeb817b7255d2868459f289ef37e563
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


class PrimitiveOp_405f672f00af24794a72fd5d594377e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 20, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_742313266a113f4c04e03b8c83de4c78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_405f672f00af24794a72fd5d594377e4
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_742313266a113f4c04e03b8c83de4c78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_405f672f00af24794a72fd5d594377e4
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


class PrimitiveOp_62b70ed1d67f5b5cba916986b9b6c55a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 12, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e1d8b3ac85302239dce725523c72c3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b70ed1d67f5b5cba916986b9b6c55a
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e1d8b3ac85302239dce725523c72c3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b70ed1d67f5b5cba916986b9b6c55a
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


class PrimitiveOp_0ff975a1a3da223d5f74635be54a1e4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 180, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4c734c9cf0755dcac80ebc4807c9c8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ff975a1a3da223d5f74635be54a1e4c
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4c734c9cf0755dcac80ebc4807c9c8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ff975a1a3da223d5f74635be54a1e4c
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class PrimitiveOp_45a1762aaa09a844c8c60a4c9ead36ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1d2d10eb450e6eeadd85cea1d1aff23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45a1762aaa09a844c8c60a4c9ead36ce
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


class PrimitiveOp_6bf684b934c1e3ec418518d26437f484(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 240, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7170dfdef1dc59a86d43969c8a30aa3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf684b934c1e3ec418518d26437f484
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7170dfdef1dc59a86d43969c8a30aa3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf684b934c1e3ec418518d26437f484
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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


class TestPrimitiveOp_872bb872ee8b8ffd51a1507c8e051f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cd4df21ac592b909f11fbe7ad66f65d
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