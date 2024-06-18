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



class PrimitiveOp_20c59d5f22fd0fe0f94ad458d3fcde66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd3d7ebd81fb78e7f7003731d89d708e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c59d5f22fd0fe0f94ad458d3fcde66
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e07824cc0acbb5006c8dac3eb7cff19d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_541457e11bd900ab98b54a46749c024b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e07824cc0acbb5006c8dac3eb7cff19d
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e5ce5ea0dcc1e86838e23e156428dc3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 160, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e865a7e90020c5a34bdf46c94785aaa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5ce5ea0dcc1e86838e23e156428dc3e
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([160, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e2be483d364eaca5fc00fa10cc689dd2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_725665734afca1d96f9f0e9af3323764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2be483d364eaca5fc00fa10cc689dd2
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_18470645a6345d0286e409214cb2dac1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c6b37891ad2eae5ff6192bbaf397f47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18470645a6345d0286e409214cb2dac1
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_663b2c3cfc48b572d2950d96100cbd52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [2, 2], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4243f14338c85a527edb8d37f80b8635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_663b2c3cfc48b572d2950d96100cbd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_216b7bf13deaa13420e3a68c83f96038(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [3, 3], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e1e7b79b211a7eea8a565ca01de4b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_216b7bf13deaa13420e3a68c83f96038
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_dfbb0025b3be6101f3eeb018aff81d18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [4, 4], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_557d4351634ff923ca0c5dc1f0355223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfbb0025b3be6101f3eeb018aff81d18
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6d7c8dc7833fdcec15fe5127b675e4d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ec5d928860f6f358ee7dfd30c6f94b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d7c8dc7833fdcec15fe5127b675e4d1
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a2df115e6580605872ac896aa89bb9a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 80, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_801d2554e4e11dff9e9478c4902fe7ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2df115e6580605872ac896aa89bb9a8
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ae819e9bff69bdb68938c5590d9e56b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 80, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_536d56c10eae94472778d626effc966b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae819e9bff69bdb68938c5590d9e56b1
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([80, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_613614edceb5e809da613cd083caee03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_508014cec324a9342adce6ff48addee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_613614edceb5e809da613cd083caee03
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f70dc7507e3bb34d29426375863d8246(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1590ba65955e211057de3dab52bdd9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f70dc7507e3bb34d29426375863d8246
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2833fde3467e1a461cda2a418ab4668b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 48, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_322d995861cca80865e0bf71f497e890(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2833fde3467e1a461cda2a418ab4668b
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([48, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_941b6ae5c3dbd6fe0e7dbed9ca43c5fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2be483d364eaca5fc00fa10cc689dd2
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9d2b611e26bc5cd5a8c1d306641969c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2332585d3b29c23ad8c2e79b3e6ed87e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d2b611e26bc5cd5a8c1d306641969c8
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f113dbf2d15fba9f348987937c111bdf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 192, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b724d03090d911ae8706e0da4de3e25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f113dbf2d15fba9f348987937c111bdf
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_28f1a036863a39c8c11b680ce75c7f72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_173204667d855d5f63fef48d11d8aeaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28f1a036863a39c8c11b680ce75c7f72
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_173204667d855d5f63fef48d11d8aeaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28f1a036863a39c8c11b680ce75c7f72
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c544f1d875e74caae61bd3cf18e4abd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [2, 2], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7d8365f28ff613540bee2f0b3bdb6b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c544f1d875e74caae61bd3cf18e4abd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b0880ff210d6c977a58ebebf78696a06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [3, 3], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_756912169fe4ede20e6cd21147658a7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0880ff210d6c977a58ebebf78696a06
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ca6245bca9ccd8d4e25acf0722d59c8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6c3c336fc7bb0b112c8044f4c2a99bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6245bca9ccd8d4e25acf0722d59c8a
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3777dcde966b44ad20a7e946247861fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [2, 2], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23027e9256446ae1b14ad6b7776a444e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3777dcde966b44ad20a7e946247861fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b048def56c501476599e5dba4f088703(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [3, 3], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_041b15af11ff97cdf3e5b43689356c07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b048def56c501476599e5dba4f088703
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3f2c2bd2724f7c3f14807e2a74d3e8eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [4, 4], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4586f63d36322baaffa1932b7c52b28b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f2c2bd2724f7c3f14807e2a74d3e8eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_aab2524f26c81fbf286eb64708d054ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 240, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c88fd1f3353730991d10f1b373fea845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aab2524f26c81fbf286eb64708d054ad
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c8a7ecdbbf8c4326923453adc40975d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 240, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf7f27a151c8c9f5be64cd55a71e534e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8a7ecdbbf8c4326923453adc40975d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_425ff85245260b54a1b1b9b77948e5e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f113dbf2d15fba9f348987937c111bdf
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5f4a121f6bcb01cd359cfff2b03da009(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 300, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eecb551690e0d1635ffb78856776a7a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f4a121f6bcb01cd359cfff2b03da009
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_496bba17a16d3bb823dd7218cf540af5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 300, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7720855ebfab5dae8b105ccccbc0b6c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_496bba17a16d3bb823dd7218cf540af5
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_64e58ddd8e98ddf0b6872efd6c8845cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 300, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc8450ef8abf2acb1d25cf385998ee91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e58ddd8e98ddf0b6872efd6c8845cf
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f5bc0d36adb5b9ffdc0e88cac7ba1147(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 300, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_30c430b6797757ce2b115e490bd78dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5bc0d36adb5b9ffdc0e88cac7ba1147
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 1, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_df69af4e3f6dc3fc24de2577ca559c18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09e4e4ef34ab331fd7582c6af40bdeab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df69af4e3f6dc3fc24de2577ca559c18
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_24407bab4b89c3c29b93695fe3fc758e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c4f858a9c6ba38035013da710bd6a8eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24407bab4b89c3c29b93695fe3fc758e
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_89ee2384ee6f24c1882ad8f338de0e06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d16446b64892b5570897aa72b63a1c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89ee2384ee6f24c1882ad8f338de0e06
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_86ca23b54e13f45b38dea12068094436(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [4, 4], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_592a27e047a74092cb80105644349413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86ca23b54e13f45b38dea12068094436
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 1, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6e0b14c79158ad411fff4a966c13a566(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [5, 5], 'EXPLICIT', 144, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48b070d34e3e24e11409c9d22e9219e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e0b14c79158ad411fff4a966c13a566
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([144, 1, 11, 11], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c8c220c127767094904d3ad1c90dd67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d2b611e26bc5cd5a8c1d306641969c8
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c7f31fa9d76940fdfab71f30e8a518a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 90, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_30de1f38d0f569f2c211945771a3160d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7f31fa9d76940fdfab71f30e8a518a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([90, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0efda94000a8803f15a852a3cfeca671(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 90, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77402915de8684fa95c4cfdc0c110017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0efda94000a8803f15a852a3cfeca671
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([90, 1, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3a52f22188b0fd951bdd73fb179fe653(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 90, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ec8685037f4b8819111733d2fea8427(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a52f22188b0fd951bdd73fb179fe653
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([90, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8d45288018748ad606953fd4909e6fdc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 90, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4119e9a5e542e7918cfb42412d6efeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d45288018748ad606953fd4909e6fdc
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([90, 1, 9, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_87f8e2ac52589507a3733ff4b265eaa6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_388b0bb6944458d951935aab8aa43067(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87f8e2ac52589507a3733ff4b265eaa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fa0e98e2b03d91bcd6b6725fcc4ca23e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [2, 2], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1fa271fbf46663e4d62096eaa216bd61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa0e98e2b03d91bcd6b6725fcc4ca23e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ee80c957f77d3a9919c1e3b255f58719(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [3, 3], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3181de2d523ea9f438c285aa4c748860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee80c957f77d3a9919c1e3b255f58719
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ce525601b2ec8482f669167051e1143f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [4, 4], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58efe17bd55ada641e6f7699f7b858cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce525601b2ec8482f669167051e1143f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c8e3f45f7e4d3979c29232b2859cc87b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 768, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9510e1b72637dd7c2c43b83c101ec143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e3f45f7e4d3979c29232b2859cc87b
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa8816f92b9e43bda04ebfb77ba14dc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e3f45f7e4d3979c29232b2859cc87b
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()