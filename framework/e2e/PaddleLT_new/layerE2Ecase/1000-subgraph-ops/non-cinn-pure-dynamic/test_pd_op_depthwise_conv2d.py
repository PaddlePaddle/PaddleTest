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


class TestPrimitiveOp_c6e1de9d298d42fc245b72b8154aaa8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_613614edceb5e809da613cd083caee03
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f060ecec9d9c4cbab3b92d1424e466f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f70dc7507e3bb34d29426375863d8246
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1, 5, 5], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_4a9f0f788f39837e13b296ea9f2c032c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2833fde3467e1a461cda2a418ab4668b
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1, 7, 7], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_6973f425b8bea80ef1faf20eb25d428d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28f1a036863a39c8c11b680ce75c7f72
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6973f425b8bea80ef1faf20eb25d428d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28f1a036863a39c8c11b680ce75c7f72
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_87d44b7dcc22b5b27fd1fcf6f397dfaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c544f1d875e74caae61bd3cf18e4abd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_bed3b17b270868b717a78836fe32326e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0880ff210d6c977a58ebebf78696a06
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e17313ae77eff5c8c6303a5abe89547f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 1280, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e9d92effc1b31433a96610d7abc2fe24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17313ae77eff5c8c6303a5abe89547f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ebd8dccd61119985e0ae11707904d16d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18470645a6345d0286e409214cb2dac1
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_1b4a92cf49643dc085edf25c7cd3b873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_663b2c3cfc48b572d2950d96100cbd52
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_895535da4d0922fc9b9eed4e19300ff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_216b7bf13deaa13420e3a68c83f96038
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_5388dec0ae254f5af5b017605412c7e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfbb0025b3be6101f3eeb018aff81d18
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


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


class TestPrimitiveOp_abbaa9b131335788739e2ba95e2d5439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c59d5f22fd0fe0f94ad458d3fcde66
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_aeca3116d657d46c4705d9921e8092b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e07824cc0acbb5006c8dac3eb7cff19d
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 1, 5, 5], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_562489fa78f37d4e86488163461093e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5ce5ea0dcc1e86838e23e156428dc3e
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 1, 7, 7], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c089b852d5c0d3966a5932b8ff619eab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f113dbf2d15fba9f348987937c111bdf
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_73c5a9e3b8160f1d45736d8629cfad1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2be483d364eaca5fc00fa10cc689dd2
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_0f164c7f860d2d0514cdd3924984ab5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d7c8dc7833fdcec15fe5127b675e4d1
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_77eacbfe83a8c9449b67284780bbb3ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2df115e6580605872ac896aa89bb9a8
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 1, 5, 5], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ad2926fd23a35155aacd68255dd9275f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae819e9bff69bdb68938c5590d9e56b1
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 1, 7, 7], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_2ed90db590c54a6fd4b3f6697b445d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f4a121f6bcb01cd359cfff2b03da009
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_21dc0962f1b123051dc028c1109d2d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_496bba17a16d3bb823dd7218cf540af5
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 1, 5, 5], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_b2c0876e6571b53d60d6b397aacab2ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e58ddd8e98ddf0b6872efd6c8845cf
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 1, 7, 7], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_811231e15cdb328e716f4dea0c44e48f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5bc0d36adb5b9ffdc0e88cac7ba1147
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 1, 9, 9], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_eee0ae76d5be7b3126b15269c55205dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d2b611e26bc5cd5a8c1d306641969c8
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_02a19d802248e8cddd2ec8750c969e2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7f31fa9d76940fdfab71f30e8a518a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([90, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_8189603714015a316908528095e6f1c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0efda94000a8803f15a852a3cfeca671
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([90, 1, 5, 5], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_66e3f0cf1074d5e048767728929c4902(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a52f22188b0fd951bdd73fb179fe653
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([90, 1, 7, 7], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_485ecc1613cd1eafc3d3f21890b959eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d45288018748ad606953fd4909e6fdc
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([90, 1, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_991f11e093cc1f891317b64b2eb45518(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2be483d364eaca5fc00fa10cc689dd2
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_449201a64d6a16979cf45ed8f9393344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df69af4e3f6dc3fc24de2577ca559c18
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ff7c5290be88aaf0fbcaeed31d4f907e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24407bab4b89c3c29b93695fe3fc758e
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 1, 5, 5], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a3487907a99dd12943af5742f5be83d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89ee2384ee6f24c1882ad8f338de0e06
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 1, 7, 7], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f68f34c0e9522c4ada576abb7f2084e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86ca23b54e13f45b38dea12068094436
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 1, 9, 9], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_b791ba270cfa323f45967fbc2fd8fdce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e0b14c79158ad411fff4a966c13a566
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 1, 11, 11], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_d6c8cfb5766a8a31e4c85dddb473afc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e3f45f7e4d3979c29232b2859cc87b
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_759372907cb338152cd48a0188f38a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17313ae77eff5c8c6303a5abe89547f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4fdecb5b297c627e2ad7a7a0ce6eba78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e3f45f7e4d3979c29232b2859cc87b
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_5bd1bebc9ec5920144c904ccb3de0899(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6245bca9ccd8d4e25acf0722d59c8a
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_60e82a5484eae407bf9ac5719b45d16e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3777dcde966b44ad20a7e946247861fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_2dd6701c2e525435f5ccc230d741fa8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b048def56c501476599e5dba4f088703
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_209c3f3d60e9ee5e32dbf8a8e627df7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f2c2bd2724f7c3f14807e2a74d3e8eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_de9589812e3147ec984b4bdc6236891f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87f8e2ac52589507a3733ff4b265eaa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_33ed20505be17573fe94104f3860bc1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa0e98e2b03d91bcd6b6725fcc4ca23e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c47679f951d914d929c2bc7a4ed1597b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee80c957f77d3a9919c1e3b255f58719
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_aa47e26aaafa5f01f14789b4c2739930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce525601b2ec8482f669167051e1143f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ae6effbe21267eb80dbfaea9d2cf561(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f113dbf2d15fba9f348987937c111bdf
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f6e9a094e62fb91241676b280a6f9567(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aab2524f26c81fbf286eb64708d054ad
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 1, 3, 3], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_dfc58bfc1ab11d578943def5e6a63699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8a7ecdbbf8c4326923453adc40975d8
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 1, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c44e3555de48c226e4613cca88d30dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d2b611e26bc5cd5a8c1d306641969c8
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()