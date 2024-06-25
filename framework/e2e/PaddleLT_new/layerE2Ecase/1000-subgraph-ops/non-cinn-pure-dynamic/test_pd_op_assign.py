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



class PrimitiveOp_bed47ae724db910ed0c662b5dfb515da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
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


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
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


class TestPrimitiveOp_eaff3dcfb5bc3b1ac35baa1844ae9586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.7415937781333923]], [[0.617002010345459]], [[0.6073431372642517]], [[0.5965418815612793]], [[0.6140335202217102]], [[0.7699944972991943]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_e187b2d1c60dd8a52196582477c6f461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.8101144433021545]], [[0.8019925951957703]], [[0.6913491487503052]], [[0.6806997060775757]], [[0.5097703337669373]], [[0.7950479388237]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad54066d4ff7755e67fe3437bd9b67c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad54066d4ff7755e67fe3437bd9b67c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad54066d4ff7755e67fe3437bd9b67c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad54066d4ff7755e67fe3437bd9b67c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ca088189e054ff6a8ab9fdb9165cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ca088189e054ff6a8ab9fdb9165cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ca088189e054ff6a8ab9fdb9165cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ca088189e054ff6a8ab9fdb9165cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
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


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_af053b1c3ac1cbb17545d70fe3589037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5fab528e924da9511a59af33fc81a057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d8b885163e0e2679bcfc7ee1ed7f123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
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


class TestPrimitiveOp_28299a65b0b8767b6d2fb8a666ae638a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([12096, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_af60f647c5b5cc5763a75597ceae0b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af60f647c5b5cc5763a75597ceae0b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af60f647c5b5cc5763a75597ceae0b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0867f4fbaec19a1d3457b3293c784bf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54b9d81c86691c557fece5fb10180706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_adf6aa74741a72bacf1210e37c7d9535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d5bfe95061403663bb7e30de62fb54ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5bfe95061403663bb7e30de62fb54ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ac95c9d1c30d5e75bcf0aed393f244b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_610cbf780e5d447441d809aed715b229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_d855079e5d9816728ff8b296ed3785f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d855079e5d9816728ff8b296ed3785f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0ac95c9d1c30d5e75bcf0aed393f244b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_27033e4ad6b6cbb24a8f3aab18d9f1c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([5376, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_df3fd6dee5f6dc5dc9b4cdc36accedf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.6954591274261475]], [[0.6292228102684021]], [[0.5073301792144775]], [[0.5909470915794373]], [[0.7551741600036621]], [[0.6076744794845581]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_fc7272626e5e7dcff4875fc00d6c045e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.7169739603996277]], [[0.6037969589233398]], [[0.6977351307868958]], [[0.5211019515991211]], [[0.5051576495170593]], [[0.6865183115005493]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_7480c6bee5e8778a040e916eaf486046(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_af053b1c3ac1cbb17545d70fe3589037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_af053b1c3ac1cbb17545d70fe3589037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54b9d81c86691c557fece5fb10180706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_3ef8b7b335ef4cedbe68e88cebe9c69b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ef8b7b335ef4cedbe68e88cebe9c69b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_af60f647c5b5cc5763a75597ceae0b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af60f647c5b5cc5763a75597ceae0b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_db930d4af0bca642e58e297715f14394(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db930d4af0bca642e58e297715f14394(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_efc2ad97b5abfa17206116fcee15c86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efc2ad97b5abfa17206116fcee15c86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0ac95c9d1c30d5e75bcf0aed393f244b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_152a0a6a2e1d8bec5e7f5a1fcc1166e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_27ed042c6705bad9c1e2b7655671250c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3412831425666809, 0.3121269941329956, -0.025228917598724365, 0.18124985694885254, 0.16395491361618042, -0.008741021156311035], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_6b864ffcf6fcc67cd18ad10e99aee5b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14869968593120575, -0.13534845411777496, -0.42750486731529236, 0.24192774295806885, -0.05786842107772827, 0.2891533374786377], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ecea7dfcb0db731ffd5ceeb5db2a54ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12379148602485657, 0.08755400776863098, -0.04719218611717224, -0.045449793338775635, 0.059109121561050415, -0.1510070264339447], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_bb7f6f6b1e8591bca43b6feb98a3812b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13721796870231628, -0.049422115087509155, -0.07374316453933716, -0.11843957006931305, 0.04739701747894287, -0.0910365879535675], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_636237153f9e814c416de9cace76fbd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05236184597015381, 0.3121269941329956, 0.0951729267835617, 0.18124985694885254, 0.17725667357444763, 0.19057394564151764], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_f0668e1c6a6224a8ab357a67589d3201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4129049479961395, 0.0738750547170639, 0.0, 0.29081887006759644, 0.20364047586917877, 0.2891533374786377], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
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


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_af053b1c3ac1cbb17545d70fe3589037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5fab528e924da9511a59af33fc81a057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d8b885163e0e2679bcfc7ee1ed7f123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_15f6190447a0daf2cbaf3cfe57fc264e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_7b9b6a713b134eea6cb01046ae4edd14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b9b6a713b134eea6cb01046ae4edd14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b9b6a713b134eea6cb01046ae4edd14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_64543cc23fb24c2a4080c6d766dfd594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5de58e6211617cb07d2bb41c855218c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5de58e6211617cb07d2bb41c855218c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5de58e6211617cb07d2bb41c855218c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54b9d81c86691c557fece5fb10180706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
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


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54b9d81c86691c557fece5fb10180706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c8966e4e0c398fed12258e1a1d4e0b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8966e4e0c398fed12258e1a1d4e0b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8966e4e0c398fed12258e1a1d4e0b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0ac95c9d1c30d5e75bcf0aed393f244b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_cbf4dcebb021b7d9826ce51ccc3b3357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf4dcebb021b7d9826ce51ccc3b3357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf4dcebb021b7d9826ce51ccc3b3357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf4dcebb021b7d9826ce51ccc3b3357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_0ac95c9d1c30d5e75bcf0aed393f244b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_5ce3d4bb02a46119535b48741eb3d058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ce3d4bb02a46119535b48741eb3d058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ce3d4bb02a46119535b48741eb3d058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ce3d4bb02a46119535b48741eb3d058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_947007dbe46f3c0d89217f6643a24a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c2fe21b86cf835065f36abcffbf895ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_690d8caa308b9bc48f37fd50e8f4e034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_690d8caa308b9bc48f37fd50e8f4e034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_690d8caa308b9bc48f37fd50e8f4e034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_690d8caa308b9bc48f37fd50e8f4e034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_690d8caa308b9bc48f37fd50e8f4e034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_673f5bea0e97f504179ed42a53640230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_673f5bea0e97f504179ed42a53640230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad54066d4ff7755e67fe3437bd9b67c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad54066d4ff7755e67fe3437bd9b67c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad54066d4ff7755e67fe3437bd9b67c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad54066d4ff7755e67fe3437bd9b67c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ca088189e054ff6a8ab9fdb9165cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ca088189e054ff6a8ab9fdb9165cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ca088189e054ff6a8ab9fdb9165cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ca088189e054ff6a8ab9fdb9165cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
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


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_17a0cf236aecb9fd1e796f445b2d913c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_727d26f03712a410d145620a657a226a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e47d4206abdbf888d5ed23c6e87fbadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e47d4206abdbf888d5ed23c6e87fbadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e47d4206abdbf888d5ed23c6e87fbadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_510a0ac790afc8fbaaceb3412de55411(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.408566951751709]], [[0.051563508808612823]], [[0.35251253843307495]], [[0.33775463700294495]], [[0.19574712216854095]], [[0.0751114934682846]], [[0.12540684640407562]], [[0.30206993222236633]], [[0.448263019323349]], [[0.23542924225330353]], [[0.36531808972358704]], [[0.14607056975364685]], [[0.39343032240867615]], [[0.07880468666553497]], [[0.23202165961265564]], [[0.3074374496936798]], [[0.33038726449012756]], [[0.3539988398551941]], [[0.40649232268333435]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
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


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
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


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
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


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_39666fbb7c9b329c55217ad21ee001d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39666fbb7c9b329c55217ad21ee001d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e47d4206abdbf888d5ed23c6e87fbadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e47d4206abdbf888d5ed23c6e87fbadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e47d4206abdbf888d5ed23c6e87fbadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7e9d5281fd7559023f6ebd70dfb9c6ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.3000502288341522]], [[0.09572094678878784]], [[0.3995654582977295]], [[0.17820869386196136]], [[0.33756303787231445]], [[0.06033669412136078]], [[0.4869387745857239]], [[0.17040975391864777]], [[0.017443474382162094]], [[0.34814712405204773]], [[0.3940105736255646]], [[0.19458813965320587]], [[0.4710569381713867]], [[0.365072101354599]], [[0.4950610101222992]], [[0.3566551208496094]], [[0.47807836532592773]], [[0.46630048751831055]], [[0.08676490187644958]], [[0.4352729916572571]], [[0.36035993695259094]]]], dtype='float32').reshape([1, 21, 1, 1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_69305d61b87735e6851966875a53e0c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_279ea1a9d525524bcd6f37b079eadc0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ac95c9d1c30d5e75bcf0aed393f244b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_456b3fca41f904dfe251bbefc8acdc4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_80b5da32786dd3c701694f472003f33b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ac95c9d1c30d5e75bcf0aed393f244b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b160dc9e5930fe846ddf986be7fdc94b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b160dc9e5930fe846ddf986be7fdc94b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54b9d81c86691c557fece5fb10180706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_011dd8a8cf9f528a29fcd3dd72d97764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011dd8a8cf9f528a29fcd3dd72d97764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
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


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0ac95c9d1c30d5e75bcf0aed393f244b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dedee686b20f791583f569238b17327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad54066d4ff7755e67fe3437bd9b67c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad54066d4ff7755e67fe3437bd9b67c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad54066d4ff7755e67fe3437bd9b67c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad54066d4ff7755e67fe3437bd9b67c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ca088189e054ff6a8ab9fdb9165cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ca088189e054ff6a8ab9fdb9165cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ca088189e054ff6a8ab9fdb9165cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ca088189e054ff6a8ab9fdb9165cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_565c4900f39d80371f28d9d342d56a8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_565c4900f39d80371f28d9d342d56a8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b65abd446bef88e236cc713531daa9b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b65abd446bef88e236cc713531daa9b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_388ddf0e4b124e3bab3c3ace6e2bcc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ba235765e794a1494f9a432a228e3d9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e710312569a9ae11fdca3a6240500d3a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_261561c46c59982f917500d3dc38190a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e23ec2d727b5ec264a52475376541410(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3f9ce11722552230610af3ae91ec5
    def get_inputs(self):
        return [
            paddle.uniform([6804, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_80fcd0c77327648aecb11a16a7495295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80fcd0c77327648aecb11a16a7495295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f51778c119ec3cc6b437a90b92097c1
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_0ac95c9d1c30d5e75bcf0aed393f244b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3b12ba443440f5e9a181ae3b079542e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86b689c5853f2c141fb0f88b97ad4cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e066b3e2d78ea5212ade52dc3309cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe605966a6f34a9324b15eeed90535c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_0ac95c9d1c30d5e75bcf0aed393f244b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_940709819bea52ae55d21c9417770125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b2500998382c6e4d65bb5e8b4b955af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
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


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e2494a6d4eb5a72efdfb1cf157582684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f479050e9e11d0f210c97dbc8239a6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245a82a18212059d61c17535c9e1104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ec9cc4e95dc08ebe80b6c76c2db500c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14fbef48c2755aabf02f1d01eaff6f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20f037ffabddcc664e0cd7500716732e
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_698f0dc65796bd84cdc501473628a3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed47ae724db910ed0c662b5dfb515da
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_1867e8239079d4d5ff8bcc2e16efec59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c9fe0b63347537ca992f17dfb72aff8
    def get_inputs(self):
        return [
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()