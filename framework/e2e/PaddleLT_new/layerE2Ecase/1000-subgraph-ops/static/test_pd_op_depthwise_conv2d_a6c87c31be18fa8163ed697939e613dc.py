import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
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



class PrimitiveOp_246d5583aeb4008de2403ec5b87e4373(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 160, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae9e5ad821a7a1a003e43659b700bf04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_246d5583aeb4008de2403ec5b87e4373
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1712ec975aeffb505f1d673efafdd88f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 160, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e678afe5ba65999563669f36ff28bff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1712ec975aeffb505f1d673efafdd88f
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b70ca5f61d8666e1a7cfa5e51a965b6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 160, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 160, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3237b196674721f0a785d0dbf250b138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b70ca5f61d8666e1a7cfa5e51a965b6f
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_499740df41873ac69521f561be3fe2c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de93e8e4bc30856cf22db1d579c0bd51
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4e967e8c96e6357bbd614fcc281337a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40b13e5abdc9e72a6c456bcfb9824e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e967e8c96e6357bbd614fcc281337a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5166da1cadcf979500bddc2e732aaecf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [2, 2], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbdfd636a9d61d6024ce91576a934b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5166da1cadcf979500bddc2e732aaecf
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_65e08f1425ee8594e9a48ae390990390(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [3, 3], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed1a26378b9deee4b2248cc7c6807ca1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65e08f1425ee8594e9a48ae390990390
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0eb51772f8bfc4cf8b12f769e475d89c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [4, 4], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_390d02dd0b6033a4265c901bddbe8ef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0eb51772f8bfc4cf8b12f769e475d89c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0714a8b0c7eeb1fac7cd93b360c7f926(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 80, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25f04b43ce48187e347850b6139ee401(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0714a8b0c7eeb1fac7cd93b360c7f926
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9717cb9437a93e533c52778dd13a483a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 80, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 80, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1453db22021d96d9bb461d9f3cbe1406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9717cb9437a93e533c52778dd13a483a
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_371b4f6ad9a9ad7546b7bb137211784d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 80, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 80, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 1, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60ab6084c317bbe8d4098554ca99e479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_371b4f6ad9a9ad7546b7bb137211784d
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e0a7c974631cc7c450ff6203c8d6e884(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 48, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0fe8d0c780a7ff59885c1bac1662e78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0a7c974631cc7c450ff6203c8d6e884
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2607e703e016cb42cb74b20550d93b8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 48, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_feccac447959a68ff142f978c8b7b885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2607e703e016cb42cb74b20550d93b8a
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1794cee7e66075229aaf75ed96331a31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 48, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 48, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 1, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9e13d0038a5015b621e3473d04bdce6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1794cee7e66075229aaf75ed96331a31
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_50ab4e599b998bb9731565522ec4e522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2853f91befd0c6757fdd392d223f9fe
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_043ace094d54d8b520bd55d112925a91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43e4e78493fb8531eed0507b7c5c9db6
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_4a66dcf0fb16d55010f28822383de2d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76b20ef6a28154e1d11cfce3d791c07e
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a488afab85a81bbe3fcfad75b6fbbe79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef7ab3dc9c448ab82e8e9073aca0108f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a488afab85a81bbe3fcfad75b6fbbe79
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef7ab3dc9c448ab82e8e9073aca0108f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a488afab85a81bbe3fcfad75b6fbbe79
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2de64174f6b646f79e98a212e0044500(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [2, 2], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_553b87ef86687300bb017e15c5937f47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de64174f6b646f79e98a212e0044500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1cc68f57eb125314e4ff813f6db55ba1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [3, 3], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1a8943309524724fbb6682ee3656707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cc68f57eb125314e4ff813f6db55ba1
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6ccafd45a166bb7b96d54b0498071bf7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c6788b8761efe25412688c739305ff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ccafd45a166bb7b96d54b0498071bf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9ae17aeea33607420ac06172c9bfe5ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [2, 2], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb43cebb128a4f8a29c2f4995891b5dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ae17aeea33607420ac06172c9bfe5ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_23872c65f40156be3f7ac581c16c0402(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [3, 3], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60097be1e8a61a380705140c660ef650(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23872c65f40156be3f7ac581c16c0402
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_13c61b840e949bf26848351f67f29a88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [4, 4], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfd02da2777b28c7c7bd77daae8ccad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13c61b840e949bf26848351f67f29a88
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_49eadcb57f372beee23fd8dd0c158ad7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 240, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 240, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_389925d4279ab5fc223f543a1e22b27c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49eadcb57f372beee23fd8dd0c158ad7
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_11db769382324b4befd8f318dd497fc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 240, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 240, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[240, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a863bef96a57702e0cd35cdb5773d26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11db769382324b4befd8f318dd497fc6
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_361f6f63862ab6c3afb8f12d0d2994cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e220cd6b83d4c8d00824906f13c698ea
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9af9f908ff8cc8b4192e1f4c7399c0fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 300, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 300, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09c413df03c8c8464b524ba12c98a911(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9af9f908ff8cc8b4192e1f4c7399c0fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b1b4b4bfb8df10109a3591abbcb83da6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 300, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 300, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57740c6da3ade18da9821fdb4b8fb277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1b4b4bfb8df10109a3591abbcb83da6
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ea002f9cf72fb860a252ab60b920a287(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 300, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 300, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 1, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e58c9c297508ef78680f1444c009dc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea002f9cf72fb860a252ab60b920a287
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_07a9318c333dc0b2c6714ef8dc386c24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 300, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 300, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 1, 9, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe6918a81e750895c32c89fb29eccdcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07a9318c333dc0b2c6714ef8dc386c24
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 1, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3301a427a11cd60c50dd10b2a50080ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 144, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc25cb40c9998f4d187d279fa41d4f4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3301a427a11cd60c50dd10b2a50080ad
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8a61c93ea32b42d28b2d9f7eda98218e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 144, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4f929d15223b219c1313e7137717e1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a61c93ea32b42d28b2d9f7eda98218e
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a29ae134c5d94a2afc030d92ee489a95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 144, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 1, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b5138552ee9ad2455a6d9c793763fdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a29ae134c5d94a2afc030d92ee489a95
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d2fc1df25dc8b9101791b172f8ce0e5a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [4, 4], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 144, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 1, 9, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_005f742fa808f62379859972c31293c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2fc1df25dc8b9101791b172f8ce0e5a
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 1, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_939333281f18de3c0d10d11ae6ebc052(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [5, 5], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 144, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[144, 1, 11, 11], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9f0e226e6c0a5ad5628b914b41c7486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_939333281f18de3c0d10d11ae6ebc052
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 1, 11, 11], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_9f857dc7640854b7fe059874af6da86d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c99f9a1997e1e4d6201f1a26297ec6f
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_42c87050bd8335849f6164fb1a1789b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 90, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 90, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[90, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7534836763c21ec73d487e4c80ba3d12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42c87050bd8335849f6164fb1a1789b4
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([90, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2da3ddde542268d87b8d6af45a3f0328(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 90, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 90, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[90, 1, 5, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fb093be3f4d2ea5213cea1f186244e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2da3ddde542268d87b8d6af45a3f0328
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([90, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f30292f9054b7cce35b3e3bc8c79c940(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 90, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 90, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[90, 1, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e9d06dfde83624f621d047422848951(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f30292f9054b7cce35b3e3bc8c79c940
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([90, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3bdd1914af19b64d27660956f4441640(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 90, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 90, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[90, 1, 9, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16bc3425df9ec41c39cb81ea903b7ff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bdd1914af19b64d27660956f4441640
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([90, 1, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_78e32f9ee72f1b99dcf3ceadaa92453f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70f71f6dc328823a63cdfe2a0e82c429(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78e32f9ee72f1b99dcf3ceadaa92453f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_baab521e96d62d59b2905a66e184fb91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [2, 2], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8e5289cc99b3db11d14d23d6e32b2dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_baab521e96d62d59b2905a66e184fb91
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_128a3aa895a412077d62f3d8db08325e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [3, 3], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ecd646cd738154bf81bb3056dab9d367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_128a3aa895a412077d62f3d8db08325e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_be4b3f6bafd8da2c1bd5a7a0dcc7976e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [4, 4], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8064d8ec52bdebfda5d3bc4679bcb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be4b3f6bafd8da2c1bd5a7a0dcc7976e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_24670f50be919bd0c02297833b84588e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71b0c0ba327986b861449c9c1dfcc377
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a3726fa07496febc088ba3c8bde4e3af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40bb1c70d2f3d1cd07a5bc0f5089e299
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()