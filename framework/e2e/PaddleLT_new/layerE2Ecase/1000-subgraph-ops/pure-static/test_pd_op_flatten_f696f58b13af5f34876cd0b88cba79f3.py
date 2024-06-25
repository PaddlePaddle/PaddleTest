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



class PrimitiveOp_660e20f534773e4324daf3c2c5e503df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_284047311aee5384e937350666a9e89c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_660e20f534773e4324daf3c2c5e503df
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c287d7be2d130efd53a9a91e955787e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ab40ab3607b4e119d3bef4319d887f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c287d7be2d130efd53a9a91e955787e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_60189029d7f22f7d47d0218b005c1cd9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 91, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e971d18a30d8c9ff85e0e85d4bf828d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60189029d7f22f7d47d0218b005c1cd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bdbe4104a1376af578c3ab1a510a2dfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 128, 16, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29c52cba70b0a3dab4ac61a2272cbe84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdbe4104a1376af578c3ab1a510a2dfd
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c2a05c8c09232bd327769d9298bcfff5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512, 256, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7bc1b62822a8f816d321eee5f76bd3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2a05c8c09232bd327769d9298bcfff5
    def get_inputs(self):
        return [
            paddle.uniform([512, 256, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_622c581c176e70ad19dc03a5a4eebb5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 84, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92fc51382d4cfcb39cb151ab4c491014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622c581c176e70ad19dc03a5a4eebb5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_35cbcab12561bf8300d242b6254d9421(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 84, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf6286587ae086196dc2b4db2450da77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35cbcab12561bf8300d242b6254d9421
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 84, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4d1ab7791efcd56f5882b41331af50a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42c99f2b63569df32400660b146cf410(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d1ab7791efcd56f5882b41331af50a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_af5a5df90c12320c495d7f14a4b203dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6807fdffd3df2efc976931dd5001850a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af5a5df90c12320c495d7f14a4b203dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e2cf32f00590c08b22a90d4ab4684152(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4364b0b0571d8884b2f9ac07b165c3ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2cf32f00590c08b22a90d4ab4684152
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1e05dc7a20ff4ad9957b0bb4159d85c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04969d3ae042be3669c38d8406639a75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e05dc7a20ff4ad9957b0bb4159d85c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_409d2d8e61ace512d89bf90af06126a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0334af7d6e528885747415ad9b230a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_409d2d8e61ace512d89bf90af06126a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7c99606390302bb2ff26cc53d38d4355(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0fc6216881c635c51ad1344b6706d09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c99606390302bb2ff26cc53d38d4355
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ba3bef83e9d7f1c14098fbd7700dd25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5669e045e68ea0cfdeca8d8ab940e0b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ba3bef83e9d7f1c14098fbd7700dd25
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_54aafe90793b1396c9681c790a2bc808(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 15, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27ca5016dbb2e398cb18eb8091caa034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54aafe90793b1396c9681c790a2bc808
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1474d3c1b52a72e75b5dff7e4b0510bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 15, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35fbded3be30063b62228d255aac640c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1474d3c1b52a72e75b5dff7e4b0510bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 15, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ef3c9e832a20b3d566ae29a0ae929f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 2), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f5b2bed6b3ba4d375b398f26254829fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ef3c9e832a20b3d566ae29a0ae929f6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
        ]


class PrimitiveOp_f4b7c3435d899ab45ff61dd6095aeb9a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_310c0bd976a4f1a9930e949c5d59dbfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4b7c3435d899ab45ff61dd6095aeb9a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int64'),
        ]


class PrimitiveOp_af806174d7b23f7be278e98b3a913e20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3faa863bb71f9f9f2361dc08ece4a00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af806174d7b23f7be278e98b3a913e20
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5098e0440a26c74d1ba3781e67a40dab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6090b74b10fd1e04928d2e773fbaa2f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5098e0440a26c74d1ba3781e67a40dab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ffcf7d4e698db00f47bbfb695d02bed3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4cb69ced9109024459d6a05431c13555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffcf7d4e698db00f47bbfb695d02bed3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8567bd5614aa07b161bc1d47abb8bd53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[128, 320, 8, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df6333e725e96626b4ec7d2d9189df04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8567bd5614aa07b161bc1d47abb8bd53
    def get_inputs(self):
        return [
            paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9a412c551cff588aeb8c6264b172fd1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46e48904935be56a19afc102265039e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a412c551cff588aeb8c6264b172fd1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6090b74b10fd1e04928d2e773fbaa2f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5098e0440a26c74d1ba3781e67a40dab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_428ee128a0c40a37d4a74073985398b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 91, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c01c0dcf92000f48d36b3c6fe62e9c3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428ee128a0c40a37d4a74073985398b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_37d66f8439b8e358c576e69bd6205268(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33c0d3e4d00bb051afc531e2b74d1a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37d66f8439b8e358c576e69bd6205268
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e11eb26113991f53d7c5d6b682e0989a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 76, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9629dbeb007464fe0aed21c45d4e82cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e11eb26113991f53d7c5d6b682e0989a
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_645e2f2365eda4bc3bb1afc1ca423285(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 2048, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d10b172024880a2c4ba471091879fb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_645e2f2365eda4bc3bb1afc1ca423285
    def get_inputs(self):
        return [
            paddle.uniform([11, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_358c88544c5ac1ab3b5e8fc3091d8602(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1000, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92bf8ea99348042aac2b7e3a76a0d877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_358c88544c5ac1ab3b5e8fc3091d8602
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_743e00354c644a49b2e8d375f89ffc30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 30, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5b8702c268b9521e8fd10e609d2a77a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743e00354c644a49b2e8d375f89ffc30
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_00d6242bc1e5c187ca4fdd9a2064456c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 30, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_125946de08f1fceff1d66beb1893414c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d6242bc1e5c187ca4fdd9a2064456c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_60714afc710764fac093f0c8b46840d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 52, 52], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d47c136f3a4a6479e4ae46bce7b443d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60714afc710764fac093f0c8b46840d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5180bf83fef2d3bd624c3808652ded3b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 52, 52], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69104ca5fe63d926637e6cbb0e63a1d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5180bf83fef2d3bd624c3808652ded3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d2ab9404c91d86dfe3adc37142b2a98e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d91c32171736d8c50b88a1897d3d0270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2ab9404c91d86dfe3adc37142b2a98e
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7634e59777277398ec86faae1a778083(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 13, 13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3410730c3abadb4b6eabfde81342ee34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7634e59777277398ec86faae1a778083
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_89419fb314c5a2a375f6538a5b9e8142(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 76, 13, 13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_adbe6c4966067b28c91222e2bd970de5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89419fb314c5a2a375f6538a5b9e8142
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a276ea90f2dac34f36a05825f9f58bb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d2acd3755f4448ab48af9b7cd1c7281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a276ea90f2dac34f36a05825f9f58bb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5cd74951b2eec9d0f36c701cb07ef979(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60b7dd00e5226d324f9697ac8379bdc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cd74951b2eec9d0f36c701cb07ef979
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_730398226ea04f50bb927275952d09de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 91, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33a808037124a644f23fb76826f5728e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_730398226ea04f50bb927275952d09de
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f231442156bf121b961fa34bd807fecb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 34, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55e8f1c794f91f5d4611acf9ed692804(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f231442156bf121b961fa34bd807fecb
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b9dbdf28806e65322252d26ecca44b15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 34, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bb408585854b4a62c3b2b75e4b710c9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9dbdf28806e65322252d26ecca44b15
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6742fba802ffae04750b8290f41cf5b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 76, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_463644900672b78ec77fe9db784ccdfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6742fba802ffae04750b8290f41cf5b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5b40ea0d271573b7b9bae5fce364b18f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 76, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa2227e4dfc75bfefb0c2125be718f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b40ea0d271573b7b9bae5fce364b18f
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 76, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fff89cea337e415763adfbea7b1f3bf0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_757bb7283a6cd54946e7c1fab5ebda46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fff89cea337e415763adfbea7b1f3bf0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04969d3ae042be3669c38d8406639a75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e05dc7a20ff4ad9957b0bb4159d85c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cac3becf0cf025d713e2166050c9440e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d29aa08dfcdf68403fa53bef29202987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cac3becf0cf025d713e2166050c9440e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_10d9905d1ce5d9e846da340fc4a84095(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cb85399d3299e73279fa8a485a5ed19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10d9905d1ce5d9e846da340fc4a84095
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a66e20450b5dd7cce321f29a27130a8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 320, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70baf4f04ec23a7b506ad9697e9822ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a66e20450b5dd7cce321f29a27130a8b
    def get_inputs(self):
        return [
            paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_22b0c3c8a0088061aa9063265bad556b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 160, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e7171c0590495a61c634c0856d7ffaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22b0c3c8a0088061aa9063265bad556b
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b4d4c3af7c23c8a3cd7a74c3fb9349be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 36, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ee6161e7226fd33ceafe213b24d8ced(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4d4c3af7c23c8a3cd7a74c3fb9349be
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a54665e0595693e5e5da954f7d46efe0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 36, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61438c5a9a1298eca4f547fc7f2150a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54665e0595693e5e5da954f7d46efe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2c43d32f452e644b222a177e6b706fbb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 36, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_291ab4cdcbac230241f8c97edcd096c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c43d32f452e644b222a177e6b706fbb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a34d18dc1aef46a584879ca099fa0622(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 64, 32, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0dd2f5256d3f64c94197081025bd232d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a34d18dc1aef46a584879ca099fa0622
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f1d43152681ed04df17ec7ec417c60e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3fd7b90c08f4570d6b263a72004e97a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1d43152681ed04df17ec7ec417c60e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46e48904935be56a19afc102265039e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a412c551cff588aeb8c6264b172fd1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_769c8159004cdb74378a170d4a5262db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 128, 16, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bdab7dc224b01d9c35ada52b9ee7accd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_769c8159004cdb74378a170d4a5262db
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3410730c3abadb4b6eabfde81342ee34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7634e59777277398ec86faae1a778083
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2ecbb1e25ae4e98852688f2549f77607(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 13, 13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b1c68d134d042be1d6ac3d945909096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ecbb1e25ae4e98852688f2549f77607
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4b9862d11dbb36084f9f93b049fceec3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[390, 64, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03607448fb2b19c02d4a7606c3abd740(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9862d11dbb36084f9f93b049fceec3
    def get_inputs(self):
        return [
            paddle.uniform([390, 64, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_71a04baf8867f56f54129da5f6ef301a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ea996c54449da77b78902c468cd0348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71a04baf8867f56f54129da5f6ef301a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa96f2bac7277f9bf5f97d4eb87708a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_826d59ac44a37fdb98e934823f143981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa96f2bac7277f9bf5f97d4eb87708a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c1076e77fc0187da7756ffa8021ab508(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 160, 16, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88857a4af1713f94f41ad566f61871bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1076e77fc0187da7756ffa8021ab508
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_71d4cc0e21f37455727ed9d2ce488828(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 2), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 768, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0b717a769c3fa4af71c30d43a0efa97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71d4cc0e21f37455727ed9d2ce488828
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9deb7738c94841e3400f3ff50c94ede6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b31ac0daef8b2dedc8f80b67e3495ef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9deb7738c94841e3400f3ff50c94ede6
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3d19675446f6004c5ac1fcf8d98f0ea4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 200, 304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e37dcbf54990587af4f4be89288ea467(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d19675446f6004c5ac1fcf8d98f0ea4
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e37dcbf54990587af4f4be89288ea467(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d19675446f6004c5ac1fcf8d98f0ea4
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ed06eab773463dd79c0d8600e1ca1c3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_301924e3cdf87df33dd5ca80a0987208(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed06eab773463dd79c0d8600e1ca1c3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_eced067de414a4387123b8540ec27713(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_224e8e271b7f5898a57d965b28c5b793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eced067de414a4387123b8540ec27713
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e25c38146b6191b0f9720d2a8faff928(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd57eb33b425752c8f6058f30540a3e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e25c38146b6191b0f9720d2a8faff928
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a0649b455f9e532c157979d51e803cfa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0951684ece2bfbe9a5e42b7c5239fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0649b455f9e532c157979d51e803cfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ab62acfeb6c5e155afdb1b192d3615bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49f9c6bd957b1e77c8d612579927272d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab62acfeb6c5e155afdb1b192d3615bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0fc6216881c635c51ad1344b6706d09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c99606390302bb2ff26cc53d38d4355
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5669e045e68ea0cfdeca8d8ab940e0b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ba3bef83e9d7f1c14098fbd7700dd25
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3de923fce09c8db030884892e6bf4a29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 21, 21], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80ee202b280766f5f8d8e86a6ad93727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3de923fce09c8db030884892e6bf4a29
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3e9afb07c485c4618bc643c8373254ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 21, 21], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2dc301e60c0e8303975605c6665ad141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e9afb07c485c4618bc643c8373254ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 21, 21], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55e8f1c794f91f5d4611acf9ed692804(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f231442156bf121b961fa34bd807fecb
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c9bddfa79525c58b0c1fd0bfc725c951(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 34, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f91580b36b0abf801c81657897f906e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9bddfa79525c58b0c1fd0bfc725c951
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7124472ad714031e775a2501dc9d4b27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 34, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa42093f2487588ed97f973b02f92f92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7124472ad714031e775a2501dc9d4b27
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5c6c3fa28ddd242837af1a3cb348cef1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1280, 32, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5cb3fec025a5ee2fac3f2fba74cd965c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c6c3fa28ddd242837af1a3cb348cef1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe35b24f2beff8ea387251275f52a3ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 256, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3dbc2a27b0319f2909f288782aa150d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe35b24f2beff8ea387251275f52a3ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4345215e96039fd139c3838e726d7eab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 18, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a65ccef31178fa1c8cc3a9fb1510d397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4345215e96039fd139c3838e726d7eab
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4709af83397f5a5ba0cfe6f83fd598fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 18, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6359667e8560944b54828ece971b408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4709af83397f5a5ba0cfe6f83fd598fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 18, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_60d259cb68af0e1e22d2700901fe7766(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 18, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_419f63d423c4b57b6162f6c05cb2e79a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60d259cb68af0e1e22d2700901fe7766
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e3cd3e5d7d6ee964b203a4a38e36b3ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 2048, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39e8dec7dee6068e7db1c928771f5fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3cd3e5d7d6ee964b203a4a38e36b3ca
    def get_inputs(self):
        return [
            paddle.uniform([43, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_37273a0896e92b49cc53d0d8496b70f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 17, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_168f6514bf61c884c681440fcadedb70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37273a0896e92b49cc53d0d8496b70f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_227d4f45c33b8f4b5deb7c75b67f36d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4dc2dd79ca7afa106e87bb12304f389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_227d4f45c33b8f4b5deb7c75b67f36d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bb82fa0f78d3f3787ee91bace7a6d3b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 17, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7a6cf5fe7033c2158229b4fedce104c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb82fa0f78d3f3787ee91bace7a6d3b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_35ece7cba2318eb4ae07de028961c945(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 100, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57afbcf9fafb80fd91f1075fddfde397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35ece7cba2318eb4ae07de028961c945
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d9c974f9e67ced854e4f8bbe5c2900e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 100, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8dd4c92917d917928327e2ef338f5e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9c974f9e67ced854e4f8bbe5c2900e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_25ea343315601db9f6c67f91181b1ec8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 96, 96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58acd4f4a31ff724520f5b096af28628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25ea343315601db9f6c67f91181b1ec8
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_32fdf423664b3519edce50e2e099427c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d011e4751fbce708f8a832fb4d7fc3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32fdf423664b3519edce50e2e099427c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f3718e8b59930c0b25b43add663df758(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f6d482dce4a205c780ef33ff3456b2a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3718e8b59930c0b25b43add663df758
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46e48904935be56a19afc102265039e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a412c551cff588aeb8c6264b172fd1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6090b74b10fd1e04928d2e773fbaa2f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5098e0440a26c74d1ba3781e67a40dab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c01c0dcf92000f48d36b3c6fe62e9c3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428ee128a0c40a37d4a74073985398b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6342ce1db4de3452cc57517167c927aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7bae1b7c22743780745db4087c439402(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6342ce1db4de3452cc57517167c927aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49f9c6bd957b1e77c8d612579927272d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab62acfeb6c5e155afdb1b192d3615bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4d297bd363c46069ba241ac95514664d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 300, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_584098ecc449a849b99faea1a26aefc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d297bd363c46069ba241ac95514664d
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f29c26e5bc9b315e6fb78cf7465d8b7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 300, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_100d1529809b5f1bd53e6aaa202dc03e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29c26e5bc9b315e6fb78cf7465d8b7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_168f6514bf61c884c681440fcadedb70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37273a0896e92b49cc53d0d8496b70f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3b45945560989f8dfe039dfead0d33a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 17, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_838a2ae935ac41264b8111bc4c0196ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b45945560989f8dfe039dfead0d33a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_903e0b046a72c2e939a35351b0d4cd17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_baa9a63e685b49b22f2a9f745bb29860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_903e0b046a72c2e939a35351b0d4cd17
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d3f6a8be632cf3d25098d38246f4a166(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 384, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8fc046f10d4dfe62a3e556e0257de6ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3f6a8be632cf3d25098d38246f4a166
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fa5908fef298b3ff47464ce1e6a8f688(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20c1128cebe14b3717c35c6c4bd08831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa5908fef298b3ff47464ce1e6a8f688
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_149041160cd6b6d610f812042db24229(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 2), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_924ed4361f88f5d9a4b9050e6b20b7f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_149041160cd6b6d610f812042db24229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[6], [6]]], dtype='int32').reshape([1, 2, 1]),
        ]


class PrimitiveOp_e6925401d6efcb6adbdc0161fc34205e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ddd11161a3645983bf9e8a450761d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6925401d6efcb6adbdc0161fc34205e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int64'),
        ]


class TestPrimitiveOp_757bb7283a6cd54946e7c1fab5ebda46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fff89cea337e415763adfbea7b1f3bf0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_910fb9ce8361ba49679ad0389123b936(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 1, 2048], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df578d1e4122192d488c5c1883af93d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_910fb9ce8361ba49679ad0389123b936
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 1, 2048], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ad17cac61dbc587b4af27014b265187a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 46, 46], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7226190ef84a8ac2d19548218bdab055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad17cac61dbc587b4af27014b265187a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_447ba5b71a2b1af0ae3436c52c32071a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 46, 46], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15938585425169fd8b2d299aa5fded4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_447ba5b71a2b1af0ae3436c52c32071a
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 46, 46], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_284047311aee5384e937350666a9e89c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_660e20f534773e4324daf3c2c5e503df
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ab40ab3607b4e119d3bef4319d887f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c287d7be2d130efd53a9a91e955787e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e971d18a30d8c9ff85e0e85d4bf828d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60189029d7f22f7d47d0218b005c1cd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_757bb7283a6cd54946e7c1fab5ebda46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fff89cea337e415763adfbea7b1f3bf0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9d9129696e1f60bf7d69ab54fb24cd41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ebd8eb3ee793e6baeece9271d69796f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d9129696e1f60bf7d69ab54fb24cd41
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2214a390be2553fcee29a21da7580042(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 2, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eefc5c0adfd7aecd059b449685f82ca5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2214a390be2553fcee29a21da7580042
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 2, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_284047311aee5384e937350666a9e89c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_660e20f534773e4324daf3c2c5e503df
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ab40ab3607b4e119d3bef4319d887f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c287d7be2d130efd53a9a91e955787e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e971d18a30d8c9ff85e0e85d4bf828d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60189029d7f22f7d47d0218b005c1cd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bc4d7491923890a8c9ac183be1b188e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0fef1d27e3f8f78d6b3f94fb26197d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc4d7491923890a8c9ac183be1b188e9
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5b2bed6b3ba4d375b398f26254829fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ef3c9e832a20b3d566ae29a0ae929f6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
        ]


class PrimitiveOp_bc2185919fb5eb69bda18f367d8f3b57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1fe8df2038ad19739d6f191a595271fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc2185919fb5eb69bda18f367d8f3b57
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int64'),
        ]


class PrimitiveOp_605068544bd95a82995063f3f9c0263f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd10c8f669c4c6a5ddac76e61e416b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_605068544bd95a82995063f3f9c0263f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_38497948e26b2b463a76a6b655d7049c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d1ceedd974f103456574fc7c55f7673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38497948e26b2b463a76a6b655d7049c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c53ef086dd53164a469a3c42a9235e51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 60, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5daa6da9be6c7eaff6edf8edc89952d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53ef086dd53164a469a3c42a9235e51
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6ee087606ce9536491993df95e5f6748(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 60, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2cd305f3ef06b1daaddeab8fd7657520(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee087606ce9536491993df95e5f6748
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0e392d13cfe7be9f2ec15e3c2d87210d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 64, 16, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_487b495b8332c9b44c7f04274a8674a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e392d13cfe7be9f2ec15e3c2d87210d
    def get_inputs(self):
        return [
            paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ab98e494d4c66ce32fcac204ff410a42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 4, 25], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67728d35daf5f2b4848c55233279793e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab98e494d4c66ce32fcac204ff410a42
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 4, 25], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d14d29354af2418995d9681395c6f1a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f292499b7b1ebdcba67c5833eb2cf03c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d14d29354af2418995d9681395c6f1a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6d482dce4a205c780ef33ff3456b2a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3718e8b59930c0b25b43add663df758
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_39df8ec83013f16a1d2b042f708d0eba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2048, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c65fa88cbabad5b97115b0b37999aafd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39df8ec83013f16a1d2b042f708d0eba
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ec3171c17e1601db5f0eb5b0402fe798(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_652f57c425e825cf4414881a06217b4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec3171c17e1601db5f0eb5b0402fe798
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c77d732d1d78a370411322e19dd8a72b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_397334a5cc2ebc8b127147eb12351b9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c77d732d1d78a370411322e19dd8a72b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_84add50e0226d033ea70abc2487c9663(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef8d14f7ba84a8f3280378c8ea51c784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84add50e0226d033ea70abc2487c9663
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d47c136f3a4a6479e4ae46bce7b443d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60714afc710764fac093f0c8b46840d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c2f5394ae73a6d0df0714f2bd8c9a4d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 76, 52, 52], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ab645d07366dbdc960301c589278a35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2f5394ae73a6d0df0714f2bd8c9a4d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe4a6f7105d5e92cb1a0aa4edf86fe51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1000, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cd722d3d2235421ecb374a2df282223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe4a6f7105d5e92cb1a0aa4edf86fe51
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0608212c806555738992e6782a02c86d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0671a9e9dfc6c82da8e6879313294fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0608212c806555738992e6782a02c86d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d45a140ec0916aceb9c0c3cffd507cdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 128, 4, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8bb8e294ba597369f38d433a1bebba4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d45a140ec0916aceb9c0c3cffd507cdd
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 4, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_284047311aee5384e937350666a9e89c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_660e20f534773e4324daf3c2c5e503df
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd10c8f669c4c6a5ddac76e61e416b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_605068544bd95a82995063f3f9c0263f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6299f7305a3b5ad1202b84946704748f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bcca9acc4bd163cdd31fad5a454d0b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6299f7305a3b5ad1202b84946704748f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_af28a4a70795c07195d298746417e07d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_144cf40d6979d310c8b3fccb3bc71452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af28a4a70795c07195d298746417e07d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3410730c3abadb4b6eabfde81342ee34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7634e59777277398ec86faae1a778083
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b1c68d134d042be1d6ac3d945909096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ecbb1e25ae4e98852688f2549f77607
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_832433196e96800de1a4d621a6d515c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 768, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7a85992709b372e4b6fcb1ac51f5136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_832433196e96800de1a4d621a6d515c6
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33c0d3e4d00bb051afc531e2b74d1a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37d66f8439b8e358c576e69bd6205268
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8b4f6ed4f91e2b3dc4093a279da576df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1e831094d4cdc1d53a9ac1a88c011b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4f6ed4f91e2b3dc4093a279da576df
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5eac8d2678c968b20b5ab353bb1da420(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 23, 23], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b2e2f989f0f47b6d2fa2d45d847ee81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5eac8d2678c968b20b5ab353bb1da420
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9c3196e6e811f39413755ff87113abac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 23, 23], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be112e3861e1c00ddd5c102c8958d387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c3196e6e811f39413755ff87113abac
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 23, 23], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46e48904935be56a19afc102265039e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a412c551cff588aeb8c6264b172fd1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6090b74b10fd1e04928d2e773fbaa2f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5098e0440a26c74d1ba3781e67a40dab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c01c0dcf92000f48d36b3c6fe62e9c3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428ee128a0c40a37d4a74073985398b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6dd05934c2634a9b5fbd38e6e0ee8fb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 256, 8, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6552104ef6631b57ccab227392721150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dd05934c2634a9b5fbd38e6e0ee8fb7
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d2acd3755f4448ab48af9b7cd1c7281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a276ea90f2dac34f36a05825f9f58bb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60b7dd00e5226d324f9697ac8379bdc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cd74951b2eec9d0f36c701cb07ef979
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33a808037124a644f23fb76826f5728e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_730398226ea04f50bb927275952d09de
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d2acd3755f4448ab48af9b7cd1c7281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a276ea90f2dac34f36a05825f9f58bb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60b7dd00e5226d324f9697ac8379bdc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cd74951b2eec9d0f36c701cb07ef979
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33a808037124a644f23fb76826f5728e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_730398226ea04f50bb927275952d09de
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_099d382f010bf13e17d116d2a3f45625(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1b2b3dd63a05931009ed43377ec19a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_099d382f010bf13e17d116d2a3f45625
    def get_inputs(self):
        return [
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1b2b3dd63a05931009ed43377ec19a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_099d382f010bf13e17d116d2a3f45625
    def get_inputs(self):
        return [
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_224e8e271b7f5898a57d965b28c5b793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eced067de414a4387123b8540ec27713
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa8bf8ef06dd43a808bcefad86de66ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aeb898334f6ac1dda0a8d25033c9501e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa8bf8ef06dd43a808bcefad86de66ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a9d7b27b53b40e8c3d9df1ce04ef8b0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_712a04c00dc1a0c09683e69a8303221c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d7b27b53b40e8c3d9df1ce04ef8b0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_179dfad6e6c820702206938092019741(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 384, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a973ea6e773bc115897f5de9bef40597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_179dfad6e6c820702206938092019741
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bee1fd93aba6faf3980bc5b70d9b8220(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af66769f393e90888735937ccad17fbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bee1fd93aba6faf3980bc5b70d9b8220
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f530071af979bf80bc495ef3a3a2ca7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 19, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4c6803973dba81c181ce071a3c075de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f530071af979bf80bc495ef3a3a2ca7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bd26ca9956ae551f5ed72f3514858a0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 19, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_538629b4de36351e547bd13a30089003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd26ca9956ae551f5ed72f3514858a0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_284047311aee5384e937350666a9e89c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_660e20f534773e4324daf3c2c5e503df
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ab40ab3607b4e119d3bef4319d887f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c287d7be2d130efd53a9a91e955787e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e971d18a30d8c9ff85e0e85d4bf828d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60189029d7f22f7d47d0218b005c1cd9
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f01b5f2a165b266566732038a1ad87fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b94a1193911762395040269e2692c118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f01b5f2a165b266566732038a1ad87fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ab40ab3607b4e119d3bef4319d887f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c287d7be2d130efd53a9a91e955787e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_321801b55b197ba9094c72aac89854af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed7369402f6d434e2bebb455465b186c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_321801b55b197ba9094c72aac89854af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dd4c92917d917928327e2ef338f5e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9c974f9e67ced854e4f8bbe5c2900e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b8754209654cb984e5bb9bc570d88e11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 2), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 768, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0903bc31eec124e7b5e7b23673b6317f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8754209654cb984e5bb9bc570d88e11
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_46e0dc05cafac923648e33c28e0449b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 256, 2, 25], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_803f9221e7ccfa08aa83e98900bd3e26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46e0dc05cafac923648e33c28e0449b5
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 2, 25], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_61d41fbc880eb5120b9a97f2e4a8df75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 136, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e73e63f6a4c375ee1ee30a0c6ada6a91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61d41fbc880eb5120b9a97f2e4a8df75
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e73e63f6a4c375ee1ee30a0c6ada6a91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61d41fbc880eb5120b9a97f2e4a8df75
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_757bb7283a6cd54946e7c1fab5ebda46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fff89cea337e415763adfbea7b1f3bf0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_100d1529809b5f1bd53e6aaa202dc03e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29c26e5bc9b315e6fb78cf7465d8b7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33c0d3e4d00bb051afc531e2b74d1a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37d66f8439b8e358c576e69bd6205268
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1e831094d4cdc1d53a9ac1a88c011b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b4f6ed4f91e2b3dc4093a279da576df
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c410c697431f813e96e67fa5892e5197(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 68, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b331431027df81de166b54ffa5ec1925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c410c697431f813e96e67fa5892e5197
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2c16d0a4beefb3c6e2818f6099066227(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 68, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_118c6cf481aed3857d3471d44ef2bf47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c16d0a4beefb3c6e2818f6099066227
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_62e22cc745faf29c680fd989bb9d5d99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 68, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_206cc2d92d76a0b2fb8666ab2123dc10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e22cc745faf29c680fd989bb9d5d99
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_65116a5ac458f2c3e89ff45e8e62fa7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 704, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8858f8cd3277c811eb0250721fb5035e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65116a5ac458f2c3e89ff45e8e62fa7b
    def get_inputs(self):
        return [
            paddle.uniform([11, 704, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_453dd83afc0b2899be2065438f854b68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_003a51d396bf7e9b14c83b69bb9d00ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_453dd83afc0b2899be2065438f854b68
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd57eb33b425752c8f6058f30540a3e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e25c38146b6191b0f9720d2a8faff928
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3c533cc2b543452a00bde938df46f131(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 72, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_750a5a65a939ba2d91c0768bc264a4d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c533cc2b543452a00bde938df46f131
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_54293f1031e14381034f635427583c4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 72, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b01d71ea3334c449060416d6d6fef7d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54293f1031e14381034f635427583c4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 72, 72], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c0a832ecd8aaa318fc40e10ab94e2b88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 72, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57f7bedd9bd987ca23e3a68aaa0a7e25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0a832ecd8aaa318fc40e10ab94e2b88
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_985fc12cc8c3bdc6ccc2409dd5f19d4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1fdd7765d6a01b3268692854ed5fa2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_985fc12cc8c3bdc6ccc2409dd5f19d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a31467e6ee071427a1bc6b3bbcf21212(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 10, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c6805ab3789c4ef3f891e0915166cfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a31467e6ee071427a1bc6b3bbcf21212
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d53ab7cc061321cbdf4e2fc3ae5683c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 10, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0d07b6f053b076d2ed152b0101b7c41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53ab7cc061321cbdf4e2fc3ae5683c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7a85992709b372e4b6fcb1ac51f5136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_832433196e96800de1a4d621a6d515c6
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b66bbfd2d2f45800cfa4c883f37b0529(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 96, 96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfcd4b8d570543a9349f000aaa0fdb4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b66bbfd2d2f45800cfa4c883f37b0529
    def get_inputs(self):
        return [
            paddle.uniform([4, 96, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_983b211a9d0bb776095e309437d4f2cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3d06a0357526683eef137ad85e9eaf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_983b211a9d0bb776095e309437d4f2cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_826d59ac44a37fdb98e934823f143981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa96f2bac7277f9bf5f97d4eb87708a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb2f2547f4150dc45b2bc0b24fc0db8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 768, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53289b12faad3cbea6aaed15a6973a6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb2f2547f4150dc45b2bc0b24fc0db8d
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04969d3ae042be3669c38d8406639a75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e05dc7a20ff4ad9957b0bb4159d85c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0334af7d6e528885747415ad9b230a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_409d2d8e61ace512d89bf90af06126a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8418bd69ad1f17a8fa6bead1b01bafeb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d55bfff127ef4301ec61f85ea8fbb819(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8418bd69ad1f17a8fa6bead1b01bafeb
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fc046f10d4dfe62a3e556e0257de6ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3f6a8be632cf3d25098d38246f4a166
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_892ac1da446cacef9e2964b5d08430ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1280, 32, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fca5cd94287ff32fbde6de27d6a685ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_892ac1da446cacef9e2964b5d08430ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53289b12faad3cbea6aaed15a6973a6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb2f2547f4150dc45b2bc0b24fc0db8d
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d47c136f3a4a6479e4ae46bce7b443d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60714afc710764fac093f0c8b46840d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69104ca5fe63d926637e6cbb0e63a1d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5180bf83fef2d3bd624c3808652ded3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d2acd3755f4448ab48af9b7cd1c7281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a276ea90f2dac34f36a05825f9f58bb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ea996c54449da77b78902c468cd0348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71a04baf8867f56f54129da5f6ef301a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e1f497631732d7c8aeacc9674e917fcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7cbb28d46153b21beaec7f80ef7a3415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1f497631732d7c8aeacc9674e917fcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e5a6a015d59a4542bd3e9a383454da2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47336416a28a12a965e30cb5a987349b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5a6a015d59a4542bd3e9a383454da2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f099c9e622559893d91d1e50ce869d83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 92, 92], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0303a56e90292441885e35926500dc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f099c9e622559893d91d1e50ce869d83
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c431935e183986dcb4c8c27cc4b3d5a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 92, 92], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8314d9a56f90418e2446fd904b15f676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c431935e183986dcb4c8c27cc4b3d5a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 92, 92], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d2acd3755f4448ab48af9b7cd1c7281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a276ea90f2dac34f36a05825f9f58bb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60b7dd00e5226d324f9697ac8379bdc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cd74951b2eec9d0f36c701cb07ef979
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33a808037124a644f23fb76826f5728e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_730398226ea04f50bb927275952d09de
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46e48904935be56a19afc102265039e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a412c551cff588aeb8c6264b172fd1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6090b74b10fd1e04928d2e773fbaa2f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5098e0440a26c74d1ba3781e67a40dab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c01c0dcf92000f48d36b3c6fe62e9c3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_428ee128a0c40a37d4a74073985398b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fc2396a310f2ff7fce67d55b1a64c98c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_836aba2dbebab5e18b8f47c12f669c63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc2396a310f2ff7fce67d55b1a64c98c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_057b7e13f926b8df7ffaf1812646a106(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07059c946e01c4b003ca3766459cd148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_057b7e13f926b8df7ffaf1812646a106
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d0f75d40750a2eb1b35d0965a111beba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36ba74c765514b6efa517f0b16f38347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0f75d40750a2eb1b35d0965a111beba
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07059c946e01c4b003ca3766459cd148(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_057b7e13f926b8df7ffaf1812646a106
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a973ea6e773bc115897f5de9bef40597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_179dfad6e6c820702206938092019741
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bdc00bd01faa6fab26d1ecbdc1de63e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac030aac63c93e395a59ebd84d51c9f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdc00bd01faa6fab26d1ecbdc1de63e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5a9059a07d96e9e3f244576c199c5e86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fbeccd2e60b909fb8860706a801ce2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9059a07d96e9e3f244576c199c5e86
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_baa9a63e685b49b22f2a9f745bb29860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_903e0b046a72c2e939a35351b0d4cd17
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fbed368be2c32cd72e39587378579720(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 42, 42], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cecd3886fd75740d6595e9d360a41478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbed368be2c32cd72e39587378579720
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0f216cf675a4a539ccc0a9de00c69a79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 42, 42], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c952a48ea117a8ceed9596bd23752e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f216cf675a4a539ccc0a9de00c69a79
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 42, 42], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20c1128cebe14b3717c35c6c4bd08831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa5908fef298b3ff47464ce1e6a8f688
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_03c95d2d27d8840add0acc94707f476d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_256809070ad2896aa4dcfce235cabb2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03c95d2d27d8840add0acc94707f476d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_741158d75c7ba6e9179be8d457761325(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c0b6b8381d55117027020ed1f061a13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_741158d75c7ba6e9179be8d457761325
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b6d482f1f5261c431ab7fb681dc68a2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 512, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd534ebc441b28c9da0ac48e608434c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d482f1f5261c431ab7fb681dc68a2f
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5978367f37b0078a5c99917d79a99bdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 512, 4, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c714512d85e899aaafe7626a34c232a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5978367f37b0078a5c99917d79a99bdd
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1efea76d7a6a585ca5a44679a0b83a4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cff5004d6f28c0248e81504543b27eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1efea76d7a6a585ca5a44679a0b83a4d
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b3fc0eed75be565642f5a08e41702ada(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 256, 8, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04185c368892f22b4d812fbd38cb741a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3fc0eed75be565642f5a08e41702ada
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b331431027df81de166b54ffa5ec1925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c410c697431f813e96e67fa5892e5197
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_29f08869b5bc087fc1f271d333a50b06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 68, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c26f396101086669883fca0702a0c448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29f08869b5bc087fc1f271d333a50b06
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0fc6216881c635c51ad1344b6706d09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c99606390302bb2ff26cc53d38d4355
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d26e68ceb51a548af3e21f9ea7642f68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3014627c9a62f582897332c58f986d9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d26e68ceb51a548af3e21f9ea7642f68
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1b9cef33bf1957226044cee488401545(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_846e358cbed9815519413d0debc9c7b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b9cef33bf1957226044cee488401545
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f058c57eb66af0dde20e6f0b20aa722a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 704, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef168ca164ef6cdf1a989526ca87e5ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f058c57eb66af0dde20e6f0b20aa722a
    def get_inputs(self):
        return [
            paddle.uniform([43, 704, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0fef1d27e3f8f78d6b3f94fb26197d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc4d7491923890a8c9ac183be1b188e9
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ebd8eb3ee793e6baeece9271d69796f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d9129696e1f60bf7d69ab54fb24cd41
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()