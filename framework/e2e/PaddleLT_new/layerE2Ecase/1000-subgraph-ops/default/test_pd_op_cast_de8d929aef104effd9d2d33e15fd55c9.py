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



class PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_45201a49f5c489e6ae24b500d9b50fc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6414e9cb5aeab69878624f5546e844fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45201a49f5c489e6ae24b500d9b50fc3
    def get_inputs(self):
        return [
            paddle.to_tensor([300.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_0e675b8c1a0954dc3434ac5f8474574b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(3549, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class PrimitiveOp_78fa2a50088b0f2c44f61257ed686456(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c45dd2f50c02131f5eaed064dea2ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7c45dd2f50c02131f5eaed064dea2ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7577a7a77dfb86679aa10e810f1de95f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56c08c19ed98775464b2d43786f1079a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d072c8f2da355d5f47c1e393e5e07496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d072c8f2da355d5f47c1e393e5e07496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_a76a192a1a376e354c649c29e2a6a604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67a4d78108e57ce95f7ea112bb7b2ac6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4096, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fbec7772e9e9acb93c9d8601776269f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fbec7772e9e9acb93c9d8601776269f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c6572c33f47d9bd3ef47b3bdef5bde37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e218a02fb81d4491a5429b99c8c6641(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16384, dtype='int32').reshape([]),
        ]


class PrimitiveOp_559f8455092a357538c194bcd01865a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_203c9727dcc034b5225cd4a2210f3e0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_559f8455092a357538c194bcd01865a7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4, 28, 28], dtype='int32'),
        ]


class TestPrimitiveOp_427726f97a88f8f83c5444ddbf7a084e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45201a49f5c489e6ae24b500d9b50fc3
    def get_inputs(self):
        return [
            paddle.to_tensor([100.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca8f1d8ab9a000f8d1de6835205a88ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_ca8f1d8ab9a000f8d1de6835205a88ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_862066ce2854c75767cd03ee0c3ac689(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2100], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4e8029bd2c8ac4e506262214022b15f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_862066ce2854c75767cd03ee0c3ac689
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
        ]


class PrimitiveOp_5caa92980d1639246e2e291c0b822d20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67824dee498aab778b691cde308e8688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5caa92980d1639246e2e291c0b822d20
    def get_inputs(self):
        return [
            paddle.to_tensor([128], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9a481d255756ea661d5c7b5617a72f8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5caa92980d1639246e2e291c0b822d20
    def get_inputs(self):
        return [
            paddle.to_tensor([16], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ab636bee62813cbd19f34501696fc38e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5caa92980d1639246e2e291c0b822d20
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_e0ac596d50708f16508305a772d679a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f593134bb4305987cffe4c58ed9a34b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ac596d50708f16508305a772d679a7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[96], dtype='int64'),
        ]


class PrimitiveOp_ae688e83c69cce732f3f0ab63bbf0637(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_129132f86ddcd6532df460861ce45ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae688e83c69cce732f3f0ab63bbf0637
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[48], dtype='int64'),
        ]


class PrimitiveOp_b96b6cfd491ffd9960c841493858128d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7fa612e6d8f774e02950956900dc446d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b96b6cfd491ffd9960c841493858128d
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype='int64').reshape([24]),
        ]


class PrimitiveOp_f821d49a59b02e9cd2f2b524b48a1510(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[12096, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec0efd95998f2a12d65d03c82b663500(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f821d49a59b02e9cd2f2b524b48a1510
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec0efd95998f2a12d65d03c82b663500(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f821d49a59b02e9cd2f2b524b48a1510
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_56c08c19ed98775464b2d43786f1079a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class PrimitiveOp_94ff6854afcbf8ddf5d94cde38a77888(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3df8c3a48a8e80cf0c3ec356b1ee2764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94ff6854afcbf8ddf5d94cde38a77888
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class PrimitiveOp_d754c6e54b0e69d20393fc36cf78f077(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c78a54aae28410c9d0cfcd8035a7bb46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d754c6e54b0e69d20393fc36cf78f077
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_18fa30e4c4319849bb6c7a0db160a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_612aa92fbaf73b16024a15b7e5eb7b87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_612aa92fbaf73b16024a15b7e5eb7b87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class PrimitiveOp_4888e346918d3535335f3532e1b10a11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4e1a1fbbba0654099b826e5113c93cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14145363867282867, 0.17830027639865875, 0.19601845741271973, 0.14128723740577698, 0.1610289216041565, 0.0832785964012146, 0.2927471101284027, 0.44636663794517517, 0.25815659761428833, 0.017593204975128174, 0.030450286343693733, 0.4521762430667877, 0.14131401479244232, 0.2590143084526062, 0.4330943524837494, 0.2948601245880127], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_8dac2ca19749ea58f5e799dc0fbb6fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
        ]


class TestPrimitiveOp_5cbd0617d5aa9e78ba192b70b5ad728c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
        ]


class TestPrimitiveOp_e47991b503a90272213e00cd0aad4118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(7581, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_b0da880982d26ce7700fc842dab18f8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(22, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d50ddb573e7e88ec00fa5202799b4fdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(6, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_00c5287fd3344da1304f2d1ca38e4041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2205b906521336d36e9ae88ab5fa1314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4725, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_00c5287fd3344da1304f2d1ca38e4041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_89cd8c15699f1de65f501ccbd74dc0cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_559f8455092a357538c194bcd01865a7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3, 28, 28], dtype='int32'),
        ]


class TestPrimitiveOp_cb61c4684a98003a1a0c0e12bf3fa301(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(577, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_5ea569821d90632c221d87d698d47aad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_911c431d1970ab28ae36db82ef2a362a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18fa30e4c4319849bb6c7a0db160a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_cdb44a5865fd80087f1e482f2e14436a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_f79359eb4197d658b4faa41dfe82264d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8009964dba10b28c5bc23d346619f77b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79359eb4197d658b4faa41dfe82264d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
        ]


class PrimitiveOp_a42bfd090c48fa83bb9538692600a70f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_536ec070d5911c4c2b797132e1e34e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a42bfd090c48fa83bb9538692600a70f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_b9e78cc7662c7e4720ee8d7f3919bd07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2a66775b01720cf7d203b3d4458efea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9e78cc7662c7e4720ee8d7f3919bd07
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 68], dtype='int32'),
        ]


class PrimitiveOp_64d7b1d64ea80364778dc57833d51d28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52d7b8b04d292383d3e7d964ebbd4fd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64d7b1d64ea80364778dc57833d51d28
    def get_inputs(self):
        return [
            paddle.uniform([1841, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4c8e727f695eb3ec720b984122c008f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c62f4659527cade01a8930c22c31ad58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8e727f695eb3ec720b984122c008f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1841, 4], dtype='int64'),
        ]


class TestPrimitiveOp_ec80dcac1f1cdf553ba46399dcc25fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0be1ad6b4d5338e9cd0643436467a247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(8400, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class PrimitiveOp_3c5bcd52d2431636790407e23aa65b90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_922a1b883e0bf4aeed6a9cbd2a2b56db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class PrimitiveOp_7c9a909507418b4175eb6ebef5673493(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_325a54cdbb69996f347272a9b3c99207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class PrimitiveOp_4bfbb0d4cd6c81b6f8cfc01ec7a6ea50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66cb74165486cb8db056065a8784ff2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bfbb0d4cd6c81b6f8cfc01ec7a6ea50
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
        ]


class PrimitiveOp_cd06dc072ea25087da606fefa6f6e8fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5376, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0aeabc8ffe23cc8b16f72e00d88fe67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd06dc072ea25087da606fefa6f6e8fd
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0aeabc8ffe23cc8b16f72e00d88fe67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd06dc072ea25087da606fefa6f6e8fd
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_00c5287fd3344da1304f2d1ca38e4041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_56c08c19ed98775464b2d43786f1079a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_0e675b8c1a0954dc3434ac5f8474574b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(3549, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class PrimitiveOp_445147eb008d0af23b194c80d911dcfa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4841c6129aa858edf2972ef49d08f081(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2284cef299175b997e6d804c0495c2ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_972bd626ff76e9c398db69fb140a2923(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_1011944269c230f9277e2cadc603d61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9566633a1edc5a11b03027789ab34973(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79359eb4197d658b4faa41dfe82264d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 4], dtype='int32'),
        ]


class TestPrimitiveOp_103a6c43bf8cbdae4a56389d0d91d860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a42bfd090c48fa83bb9538692600a70f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9a51d61fb37210c90eba83dfe0bb250f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9e78cc7662c7e4720ee8d7f3919bd07
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 68], dtype='int32'),
        ]


class TestPrimitiveOp_00c046bd60945d940e324393266b480f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64d7b1d64ea80364778dc57833d51d28
    def get_inputs(self):
        return [
            paddle.uniform([5562, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1367a5e6763e091a3da25304a700dcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8e727f695eb3ec720b984122c008f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5562, 4], dtype='int64'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class PrimitiveOp_c0b9fcdb09a831c89edd47d77e85fd3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7aba3b0d6dc5b0eede56d387badbc9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0b9fcdb09a831c89edd47d77e85fd3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5c36366e9691196fd50d3c8fec28e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c22d5924397240bc766dba7ead695b75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(10, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_dab03ec3227b6db5910026c1af60e06b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7e8db426915121c7707c7adba460804f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(98, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3dea7598069e5d4b2ab7f081a302b9c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(99, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_5fad9e1aee57e5b9bd950cbd302aa9b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66debda542f81ca4387c9f7360b95451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
        ]


class TestPrimitiveOp_66debda542f81ca4387c9f7360b95451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_9ebf0cf29337c314ccf94b913a219ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(192, dtype='int32').reshape([]),
        ]


class PrimitiveOp_7411643bcd1da98ac47335bed3727a59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b862043eefe4c828d4d19ff7cca852ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411643bcd1da98ac47335bed3727a59
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4775af02a2c76cdc92c7dc5fc103655e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 192, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d50ddb573e7e88ec00fa5202799b4fdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(6, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_b5b17b615705d618b467a4f7a59600dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d754c6e54b0e69d20393fc36cf78f077
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_b0da880982d26ce7700fc842dab18f8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(22, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18fa30e4c4319849bb6c7a0db160a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d50ddb573e7e88ec00fa5202799b4fdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(6, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_4841c6129aa858edf2972ef49d08f081(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_972bd626ff76e9c398db69fb140a2923(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d50ddb573e7e88ec00fa5202799b4fdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(6, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c22d5924397240bc766dba7ead695b75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(10, dtype='int32').reshape([]),
        ]


class PrimitiveOp_b8cd3eac4b0759bcf033ac4911afa05c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_417159ee557ba01b15c6f00ead299789(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8cd3eac4b0759bcf033ac4911afa05c
    def get_inputs(self):
        return [
            paddle.to_tensor([False, False, False, False, False, True], dtype='bool').reshape([6]),
        ]


class TestPrimitiveOp_acbcbcebfcb12fc277cf179be8629ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8cd3eac4b0759bcf033ac4911afa05c
    def get_inputs(self):
        return [
            paddle.to_tensor([False, False, False, False, False, False], dtype='bool').reshape([6]),
        ]


class TestPrimitiveOp_cdb44a5865fd80087f1e482f2e14436a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_8009964dba10b28c5bc23d346619f77b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79359eb4197d658b4faa41dfe82264d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
        ]


class TestPrimitiveOp_536ec070d5911c4c2b797132e1e34e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a42bfd090c48fa83bb9538692600a70f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_8d193ac3f030a312eecb2c761806a99b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 76], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab47c319de14616e1ae85556d01c6ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d193ac3f030a312eecb2c761806a99b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 76], dtype='int32'),
        ]


class TestPrimitiveOp_f47d74427c84b099137a60b799edae71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64d7b1d64ea80364778dc57833d51d28
    def get_inputs(self):
        return [
            paddle.uniform([1734, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d5ce935b754768d0165032fbc576b2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8e727f695eb3ec720b984122c008f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1734, 4], dtype='int64'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class PrimitiveOp_c6cdf8fd05410c60030318097a38bed0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_888ca738bc97c89f837862f07414f7ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6cdf8fd05410c60030318097a38bed0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d440325133ced06c0860e0e045b518c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18058c7d8030fb70149a48c8c39280f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6cdf8fd05410c60030318097a38bed0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d440325133ced06c0860e0e045b518c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b0da880982d26ce7700fc842dab18f8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(22, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_9ffd376709422bb3f4689546b57e19fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(28, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_ecbbfc83d253a4ce3ebf02552468b1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(50, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3b8d9a8166c0f8fcf64220b6a289d791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_83587564bf21ee06010ed91e956f8a93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4116, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_468cb29e0a36fff21e529376dcdd426d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab1ce2d22110e2915ecdb583b83d796c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_6f83ead9f1391547f428b411ea7cc4e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d934cccfd43a47e80830afeb3be06a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f83ead9f1391547f428b411ea7cc4e5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[80], dtype='int64'),
        ]


class PrimitiveOp_0b0c73d4c256c4255c2b4fb65f43dd95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_187bb2da737cf70bbc61d25222906792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b0c73d4c256c4255c2b4fb65f43dd95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
        ]


class PrimitiveOp_3b21bfbfb24128ef45526e251e7c4e58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd9f764bc3db097b9808ed2ee2a0b250(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b21bfbfb24128ef45526e251e7c4e58
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype='int64').reshape([20]),
        ]


class PrimitiveOp_2d6befd9b4a95789b5118b1d0700bf47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c31cf99ec009e21a13045b90c9873b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d6befd9b4a95789b5118b1d0700bf47
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c31cf99ec009e21a13045b90c9873b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d6befd9b4a95789b5118b1d0700bf47
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ecc33a433f965f8db57958bec06e0998(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59343fcd9a9dbfa44b84de2b46d6280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecc33a433f965f8db57958bec06e0998
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_56c08c19ed98775464b2d43786f1079a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_255896d63ab8ad873df1fce041c67636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10497827082872391, 0.24310307204723358, 0.481964647769928, 0.0501120500266552, 0.17494754493236542, 0.34785979986190796, 0.41855791211128235, 0.3518923223018646, 0.05826127529144287, 0.19665104150772095, 0.3200782537460327, 0.37988394498825073, 0.2087623029947281, 0.45571592450141907, 0.08767567574977875, 0.0012626966927200556, 0.2798956632614136, 0.43433037400245667, 0.4705473780632019, 0.01220680121332407, 0.16760872304439545, 0.1691567450761795, 0.4557212293148041, 0.361501008272171], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_f1bab6b0b640713b4a3bcc00ab17b2f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
        ]


class TestPrimitiveOp_f287a511eabbc1558de72998e1503fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
        ]


class PrimitiveOp_fa66f160a3e34528e0d691b1ea16f430(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d43b8da7338a0dbb59b4e8a2a85e20a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa66f160a3e34528e0d691b1ea16f430
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29211392998695374], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7c45dd2f50c02131f5eaed064dea2ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7c45dd2f50c02131f5eaed064dea2ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_7577a7a77dfb86679aa10e810f1de95f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56c08c19ed98775464b2d43786f1079a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d072c8f2da355d5f47c1e393e5e07496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d072c8f2da355d5f47c1e393e5e07496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_a76a192a1a376e354c649c29e2a6a604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67a4d78108e57ce95f7ea112bb7b2ac6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4096, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fbec7772e9e9acb93c9d8601776269f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fbec7772e9e9acb93c9d8601776269f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c6572c33f47d9bd3ef47b3bdef5bde37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e218a02fb81d4491a5429b99c8c6641(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16384, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_0af65c9bca4959c9c8a0974f06d5a4b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(6069, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_85e11fc25207591dacc37fef83270607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_fdce960a18d2872109a101168450d07b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79359eb4197d658b4faa41dfe82264d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 4], dtype='int32'),
        ]


class TestPrimitiveOp_40cbebed856c3df350215725e69f2b08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a42bfd090c48fa83bb9538692600a70f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_185fb007383cd915860df5d3d8f07612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9e78cc7662c7e4720ee8d7f3919bd07
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 68], dtype='int32'),
        ]


class TestPrimitiveOp_0b27f1ee4e525619a74ae006b8f79d09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64d7b1d64ea80364778dc57833d51d28
    def get_inputs(self):
        return [
            paddle.uniform([1541, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5752ef8a998fffb52628619978d874f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8e727f695eb3ec720b984122c008f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1541, 4], dtype='int64'),
        ]


class TestPrimitiveOp_cdb44a5865fd80087f1e482f2e14436a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_cdb44a5865fd80087f1e482f2e14436a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_7b9cd4e833f1251903b6b017a7ff000a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3549], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00f223b5e1dbab64be8787c781e0c0ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b9cd4e833f1251903b6b017a7ff000a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
        ]


class TestPrimitiveOp_072c21148abece54a46afccda2b5a166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecc33a433f965f8db57958bec06e0998
    def get_inputs(self):
        return [
            paddle.to_tensor([8, 2], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18fa30e4c4319849bb6c7a0db160a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7f25ba8fba52d23f7713fd3c9b4b678c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20741339027881622, 0.24170714616775513, 0.16768890619277954, 0.05824856087565422], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_8bbdca277bfb57d348a1afe47e48507a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f0d1762971dbbc8d54f7af458b2fd659(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_00c5287fd3344da1304f2d1ca38e4041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_0891190351b0d63b1ab1cbbb773d9a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(52, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_15c1cba823987cec165bd51ada779fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(202, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_947efb18bf71d7887091f78c13ccb9d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1025, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class PrimitiveOp_c0c339307df077ee29ceadc7a67febc0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61cabe75e54f2ca0b969456f476c86aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0c339307df077ee29ceadc7a67febc0
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype='int64').reshape([14]),
        ]


class PrimitiveOp_4dcfd9c76b3131d7e69112c43e97f12f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98d07464f6864667253a8115d90213de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dcfd9c76b3131d7e69112c43e97f12f
    def get_inputs(self):
        return [
            paddle.uniform([14, 14, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_968db2a10050de795f41133dbea6ede4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69e6c521d3f5750a53b2082273e61341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_968db2a10050de795f41133dbea6ede4
    def get_inputs(self):
        return [
            paddle.uniform([14, 14, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3d6ab6bf4351f6379abbd0bc87da57da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f497de691327ccec97cac470cfe6117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6ab6bf4351f6379abbd0bc87da57da
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype='int64').reshape([28]),
        ]


class PrimitiveOp_452e1056fb286a27a6929f62b56a2e98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_323548b532cafa54ed04c7f984fec0b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_452e1056fb286a27a6929f62b56a2e98
    def get_inputs(self):
        return [
            paddle.uniform([28, 28, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2bf9a323b46d688b520f5cd1c6534e13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4be1252541de9165ddd4a115584389a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf9a323b46d688b520f5cd1c6534e13
    def get_inputs(self):
        return [
            paddle.uniform([28, 28, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b7c1ac54a35b296964fcc1e017e75296(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76273dc199b468bf3e6c61c0272c6405(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7c1ac54a35b296964fcc1e017e75296
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[56], dtype='int64'),
        ]


class PrimitiveOp_ee045f2bcaadac07b467208c83b86f68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_067ed7a10b5f2cca602945de7daca89c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee045f2bcaadac07b467208c83b86f68
    def get_inputs(self):
        return [
            paddle.uniform([56, 56, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9788c8e80ce7a1c98202ccbe608d036e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b68ffeba3304e847bc7de977befe4c64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9788c8e80ce7a1c98202ccbe608d036e
    def get_inputs(self):
        return [
            paddle.uniform([56, 56, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18fa30e4c4319849bb6c7a0db160a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_612aa92fbaf73b16024a15b7e5eb7b87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_612aa92fbaf73b16024a15b7e5eb7b87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2603c1f032f45d46760d00391f61084e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(104, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2603c1f032f45d46760d00391f61084e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(104, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7911037999e5f9d9846dbf02fe637b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7911037999e5f9d9846dbf02fe637b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_dcb9a3102d70e0e89f901ab32300621c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4116], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93e36ea4849ae9ebc6ac1ff7e0c91f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcb9a3102d70e0e89f901ab32300621c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class PrimitiveOp_34edc944926fad1beb203880b60b3548(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_968983cc2d72c71ab8cf34af13e715dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34edc944926fad1beb203880b60b3548
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_7fe9d49f358a8f6c92d330f60d8b5c0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34edc944926fad1beb203880b60b3548
    def get_inputs(self):
        return [
            paddle.to_tensor(7, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_56c08c19ed98775464b2d43786f1079a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6414e9cb5aeab69878624f5546e844fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45201a49f5c489e6ae24b500d9b50fc3
    def get_inputs(self):
        return [
            paddle.to_tensor([300.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_81e92b6774203d74c0434b566b8f498c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b8d9a8166c0f8fcf64220b6a289d791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7911037999e5f9d9846dbf02fe637b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_2f4e5c9d90f968a3a1ce3bb082565cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79359eb4197d658b4faa41dfe82264d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
        ]


class TestPrimitiveOp_00e28804eac776c419c249adcdede230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a42bfd090c48fa83bb9538692600a70f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_deb804c583baa286be9e9fc9ea8b54e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9e78cc7662c7e4720ee8d7f3919bd07
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
        ]


class TestPrimitiveOp_2d31b3238bf65b4310985f967c38ea03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64d7b1d64ea80364778dc57833d51d28
    def get_inputs(self):
        return [
            paddle.uniform([2061, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b209b8b4b0d9889ac91c6b28274bef07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8e727f695eb3ec720b984122c008f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2061, 4], dtype='int64'),
        ]


class TestPrimitiveOp_00c5287fd3344da1304f2d1ca38e4041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18fa30e4c4319849bb6c7a0db160a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_aea740c893bb085170af9ca8556f1b78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(14, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2cfd29fba9326bececfff718e0ded98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(25, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18fa30e4c4319849bb6c7a0db160a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18fa30e4c4319849bb6c7a0db160a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_5c7032b06f00176674606527880e09a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_b81237f634ed9e178c623ec48deb2639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79359eb4197d658b4faa41dfe82264d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 4], dtype='int32'),
        ]


class TestPrimitiveOp_bbabeb0ca0553d1a03e0967d33e3fa22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a42bfd090c48fa83bb9538692600a70f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_69f921e9f6767e44beb85a203b62ead9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9e78cc7662c7e4720ee8d7f3919bd07
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 68], dtype='int32'),
        ]


class TestPrimitiveOp_b0818880c4404eb365a6e7bac21197ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64d7b1d64ea80364778dc57833d51d28
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5e7885538440097c4324ae4b68d524b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8e727f695eb3ec720b984122c008f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4642, 4], dtype='int64'),
        ]


class TestPrimitiveOp_9dc91178ed779914590bd3184e12bf51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_559f8455092a357538c194bcd01865a7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[6, 28, 28], dtype='int32'),
        ]


class TestPrimitiveOp_7fd498bfb67e9c7d5ad0175c5ae42b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94ff6854afcbf8ddf5d94cde38a77888
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_ca8f1d8ab9a000f8d1de6835205a88ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_a1c4c34d0e7978caaa6756be9810c2f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79359eb4197d658b4faa41dfe82264d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 4], dtype='int32'),
        ]


class TestPrimitiveOp_19de09d5908e1986e76e4d1f63776be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a42bfd090c48fa83bb9538692600a70f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9399d5f07890f60678169a26f52674ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9e78cc7662c7e4720ee8d7f3919bd07
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 68], dtype='int32'),
        ]


class TestPrimitiveOp_d71e6d668409bc7e444fc450ab60444b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64d7b1d64ea80364778dc57833d51d28
    def get_inputs(self):
        return [
            paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b78cde9f594c79f3911c84adc5cbae2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8e727f695eb3ec720b984122c008f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1042, 4], dtype='int64'),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_cb1ab52c6c33bdf7ade722c9dc8c5d3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(9261, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c22d5924397240bc766dba7ead695b75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(10, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18fa30e4c4319849bb6c7a0db160a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


class PrimitiveOp_76b84f38a3d8917252bbaf22dc1d3d5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e78b0a39988f3256999be59539e37889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76b84f38a3d8917252bbaf22dc1d3d5d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[68], dtype='int64'),
        ]


class PrimitiveOp_04a7407169a437cb91a8f97b937d48f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_41bad948af63f8df2d524a800d2667c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04a7407169a437cb91a8f97b937d48f4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[34], dtype='int64'),
        ]


class PrimitiveOp_d0d5dfaa8b40c4f51bd48928b16cb385(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2eea0685801d846081a1b8923d36b46f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0d5dfaa8b40c4f51bd48928b16cb385
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype='int64').reshape([17]),
        ]


class PrimitiveOp_3fd1cdc55631f9dbe35be20b9881759f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_073687fbda6abf362bd6007735f05989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fd1cdc55631f9dbe35be20b9881759f
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_073687fbda6abf362bd6007735f05989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fd1cdc55631f9dbe35be20b9881759f
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7aba3b0d6dc5b0eede56d387badbc9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0b9fcdb09a831c89edd47d77e85fd3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5c36366e9691196fd50d3c8fec28e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7aba3b0d6dc5b0eede56d387badbc9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0b9fcdb09a831c89edd47d77e85fd3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5c36366e9691196fd50d3c8fec28e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7aba3b0d6dc5b0eede56d387badbc9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0b9fcdb09a831c89edd47d77e85fd3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5c36366e9691196fd50d3c8fec28e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c593e8c6b6abeae488761917b767345b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(2048, dtype='int32').reshape([]),
        ]


class PrimitiveOp_4c0693de956e8ad45bd0c1785d6c3727(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_853ff5456420067c561a668a75445ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c0693de956e8ad45bd0c1785d6c3727
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6d67d83938fa17ff0c9e06838e145c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fbec7772e9e9acb93c9d8601776269f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fbec7772e9e9acb93c9d8601776269f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c6572c33f47d9bd3ef47b3bdef5bde37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e218a02fb81d4491a5429b99c8c6641(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16384, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d072c8f2da355d5f47c1e393e5e07496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d072c8f2da355d5f47c1e393e5e07496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_a76a192a1a376e354c649c29e2a6a604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67a4d78108e57ce95f7ea112bb7b2ac6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4096, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7c45dd2f50c02131f5eaed064dea2ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7c45dd2f50c02131f5eaed064dea2ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_7577a7a77dfb86679aa10e810f1de95f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56c08c19ed98775464b2d43786f1079a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_56d5378017754b901a36c34d3a89ed40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_56d5378017754b901a36c34d3a89ed40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
        ]


class TestPrimitiveOp_69b1e175c0d0ce5508754c2018c965fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([16, 16, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_00c5287fd3344da1304f2d1ca38e4041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6582ede0a83c426599075291239aa7b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
        ]


class TestPrimitiveOp_00c5287fd3344da1304f2d1ca38e4041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6582ede0a83c426599075291239aa7b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
        ]


class TestPrimitiveOp_52d654544bd178f60a029666b9b2db9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([8, 8, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_dedfca3cd72c574c8c9ba49102515320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(2100, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_4ca37ed42a51c779fbbd29b45c3219a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0b9fcdb09a831c89edd47d77e85fd3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5c36366e9691196fd50d3c8fec28e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_4ca37ed42a51c779fbbd29b45c3219a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0b9fcdb09a831c89edd47d77e85fd3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5c36366e9691196fd50d3c8fec28e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_4ca37ed42a51c779fbbd29b45c3219a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0b9fcdb09a831c89edd47d77e85fd3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5c36366e9691196fd50d3c8fec28e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c593e8c6b6abeae488761917b767345b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(2048, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_764fc6ed4f46143f6f82b9e80eca4fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c0693de956e8ad45bd0c1785d6c3727
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6d67d83938fa17ff0c9e06838e145c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_947efb18bf71d7887091f78c13ccb9d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1025, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7856772ebd11dbd2ef8d56ee395b28d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_c5797905b2356d4dd6c00f60be7a1aa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79359eb4197d658b4faa41dfe82264d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 4], dtype='int32'),
        ]


class TestPrimitiveOp_b6d88eaf1d408d196fd5f38f5a9936c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a42bfd090c48fa83bb9538692600a70f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_13f208bf5fd233681ac2ef578134961c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9e78cc7662c7e4720ee8d7f3919bd07
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 68], dtype='int32'),
        ]


class TestPrimitiveOp_652c5c2954d807d195a589611c5f9681(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64d7b1d64ea80364778dc57833d51d28
    def get_inputs(self):
        return [
            paddle.uniform([2369, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d27991782f6f923bd3656ed764ed02f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8e727f695eb3ec720b984122c008f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2369, 4], dtype='int64'),
        ]


class TestPrimitiveOp_b0da880982d26ce7700fc842dab18f8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(22, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_f1a150d5ccc6ab419104f692abc582f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_ef60c0659b7f5ea3ecfe828cfcb681d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79359eb4197d658b4faa41dfe82264d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 4], dtype='int32'),
        ]


class TestPrimitiveOp_3485397eecbdb6421bd152208145f099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a42bfd090c48fa83bb9538692600a70f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_ab83989a4fbb399d3c9e320936adff14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9e78cc7662c7e4720ee8d7f3919bd07
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 68], dtype='int32'),
        ]


class TestPrimitiveOp_53aa5bde9937706657e1a78aefd6bc9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64d7b1d64ea80364778dc57833d51d28
    def get_inputs(self):
        return [
            paddle.uniform([3054, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36e874a3a4c0fb64e68462c3cb47c9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8e727f695eb3ec720b984122c008f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3054, 4], dtype='int64'),
        ]


class TestPrimitiveOp_1d29f0b94456689b651ffc6140f864ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d020bdeb6bbeff62efa560703457e2f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79359eb4197d658b4faa41dfe82264d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 4], dtype='int32'),
        ]


class TestPrimitiveOp_31f80d341fd232c387f349a5f58228e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a42bfd090c48fa83bb9538692600a70f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_ed3f6617dca5085966759367d2afa9f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9e78cc7662c7e4720ee8d7f3919bd07
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 68], dtype='int32'),
        ]


class TestPrimitiveOp_c35c91fb2f750fb1d4ecef5eea0ad8ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64d7b1d64ea80364778dc57833d51d28
    def get_inputs(self):
        return [
            paddle.uniform([3819, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_adc187119fdd7eb557e11f7d5332d960(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8e727f695eb3ec720b984122c008f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3819, 4], dtype='int64'),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_427726f97a88f8f83c5444ddbf7a084e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45201a49f5c489e6ae24b500d9b50fc3
    def get_inputs(self):
        return [
            paddle.to_tensor([100.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_56c08c19ed98775464b2d43786f1079a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_9ecbc8793eef7aacf7b96ad673d1f1a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11109, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7c4a8f870ab1df6757710a20d9cd82d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_72042c4da4da7c07b4def7bb42cada39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_559f8455092a357538c194bcd01865a7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2, 28, 28], dtype='int32'),
        ]


class TestPrimitiveOp_e0c21bbd2c938245a284ba0c87a4f5c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5caa92980d1639246e2e291c0b822d20
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_c9334bff279bfe92692530fbe3a3fc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5caa92980d1639246e2e291c0b822d20
    def get_inputs(self):
        return [
            paddle.to_tensor([11], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_535cf105551cb18a07f7edd3c15f47cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5caa92980d1639246e2e291c0b822d20
    def get_inputs(self):
        return [
            paddle.to_tensor([384], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9bb80125bc6c7db5a4e796922a28bdbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5caa92980d1639246e2e291c0b822d20
    def get_inputs(self):
        return [
            paddle.to_tensor([28], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_569e993c4e6d8fe53a7d4f5a38ba07be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5caa92980d1639246e2e291c0b822d20
    def get_inputs(self):
        return [
            paddle.to_tensor([77], dtype='int32').reshape([1]),
        ]


class PrimitiveOp_799b0b481eb4c0091019454328b2472e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[152], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb8f824c2b02057953295128bd4d53a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_799b0b481eb4c0091019454328b2472e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[152], dtype='int64'),
        ]


class PrimitiveOp_e255e6c23bc992966d1077b792872ffe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dffe1bbb23bf25f05f796b2af673415b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e255e6c23bc992966d1077b792872ffe
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[100], dtype='int64'),
        ]


class PrimitiveOp_7d4d294cc65230dc31138897576889e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e831d294de30fefa30a81a0911cd07cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d4d294cc65230dc31138897576889e2
    def get_inputs(self):
        return [
            paddle.uniform([100, 152, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_43d2434285ebdf31b3024c71f6c9bb30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa28fe19ffd14d34aa4bc6f3a1382c6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43d2434285ebdf31b3024c71f6c9bb30
    def get_inputs(self):
        return [
            paddle.uniform([100, 152, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8ebbb6f7a10d61bf9d06173a9e0b390d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[76], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89a3af0633516e286ec714668eda1ad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ebbb6f7a10d61bf9d06173a9e0b390d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[76], dtype='int64'),
        ]


class PrimitiveOp_2de07cf84782a2c250cadea55b33c3e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7df39f33c8899ce58104e0bad72efb12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de07cf84782a2c250cadea55b33c3e8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[50], dtype='int64'),
        ]


class PrimitiveOp_4d8185a205fd4fcb2709b9475ef0dba5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c97bdd25cbb4d3694ae54b7ec48b51b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d8185a205fd4fcb2709b9475ef0dba5
    def get_inputs(self):
        return [
            paddle.uniform([50, 76, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_16a07c82005c2fe4768234e5f8e5ff96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c86d8d3438624d5244e21d9035084ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16a07c82005c2fe4768234e5f8e5ff96
    def get_inputs(self):
        return [
            paddle.uniform([50, 76, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb9118d1fabc8c15e0c2ff56d9046a64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a20c94f2b3ebc5cbf2b5991774af513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9118d1fabc8c15e0c2ff56d9046a64
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[38], dtype='int64'),
        ]


class PrimitiveOp_02ca4948404d699f3ab456a1ac020480(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92142c1c7b5bdfb52fc3770c444fead4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02ca4948404d699f3ab456a1ac020480
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype='int64').reshape([25]),
        ]


class PrimitiveOp_068f821a63f0db56f5e6d37a6b0527e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_639e1d7a757303840572a713646bd294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_068f821a63f0db56f5e6d37a6b0527e0
    def get_inputs(self):
        return [
            paddle.uniform([25, 38, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3ee7bd93bccb1d8c6ddd320b4f6023d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88377d3fb05dc0110122361a081c8863(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ee7bd93bccb1d8c6ddd320b4f6023d0
    def get_inputs(self):
        return [
            paddle.uniform([25, 38, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_80990823450ceb2d17c0920c6454ec93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[19], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f77e749e5c3287c22ac205e547578d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80990823450ceb2d17c0920c6454ec93
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype='int64').reshape([19]),
        ]


class PrimitiveOp_a3cb29218907de5b58b704467f95018e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22562f47e74f06c0b277a1fa9474c13d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3cb29218907de5b58b704467f95018e
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype='int64').reshape([13]),
        ]


class PrimitiveOp_19f8b21573be9decf6a8a665c454b4c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c427b13b150f62b9a94e0ccbc82ecd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19f8b21573be9decf6a8a665c454b4c1
    def get_inputs(self):
        return [
            paddle.uniform([13, 19, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bd63c9a58f931c1d4b2aacd64f9f7c3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58d737efa051236527ea7d2ca95e5fb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd63c9a58f931c1d4b2aacd64f9f7c3c
    def get_inputs(self):
        return [
            paddle.uniform([13, 19, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9f0e60e72f075637eb4c0bf463bf4d32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56965883fd0fad1cc68f1a59723a0588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f0e60e72f075637eb4c0bf463bf4d32
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64').reshape([10]),
        ]


class PrimitiveOp_b7a01867df85272a9f8d629403e1c948(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af27e2c9e8e8b95e43bfcda6db0b684f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7a01867df85272a9f8d629403e1c948
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6], dtype='int64').reshape([7]),
        ]


class PrimitiveOp_d12de5600cf7096b5d99c04f1590e4e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8ee80cb0dd75889826acfbdaddf1741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d12de5600cf7096b5d99c04f1590e4e9
    def get_inputs(self):
        return [
            paddle.uniform([7, 10, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_eba2d2e57834654a378d577af55d375f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2981ebaf9e1d6202dc09f6c6ce873356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eba2d2e57834654a378d577af55d375f
    def get_inputs(self):
        return [
            paddle.uniform([7, 10, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class PrimitiveOp_04dd84297eaf82fac5b404b751288e23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee2d05b7932facf71339d3a1ccf9cc07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04dd84297eaf82fac5b404b751288e23
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab1ce2d22110e2915ecdb583b83d796c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7d268f10609b091c87d40baac21d920c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6cdf8fd05410c60030318097a38bed0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d440325133ced06c0860e0e045b518c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d50ddb573e7e88ec00fa5202799b4fdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(6, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_1f353eaaa70a480b40b335207db6fb3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.to_tensor([0.059932973235845566, 0.1609245091676712, 0.4488435387611389, 0.004289453383535147, 0.1617693454027176, 0.06451651453971863, 0.2686924338340759, 0.12058482319116592, 0.1321176439523697, 0.3679129481315613, 0.2524234354496002, 0.23021575808525085, 0.298456072807312, 0.45019611716270447, 0.49008193612098694, 0.017989452928304672, 0.4127422571182251, 0.24961227178573608, 0.19512465596199036, 0.470866858959198], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_7268aa07933dab324897a95f7aabcece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
        ]


class TestPrimitiveOp_c11b327b5370b1d5b7dc5cee55275784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7c45dd2f50c02131f5eaed064dea2ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7c45dd2f50c02131f5eaed064dea2ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_7577a7a77dfb86679aa10e810f1de95f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56c08c19ed98775464b2d43786f1079a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d072c8f2da355d5f47c1e393e5e07496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d072c8f2da355d5f47c1e393e5e07496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_a76a192a1a376e354c649c29e2a6a604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67a4d78108e57ce95f7ea112bb7b2ac6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4096, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fbec7772e9e9acb93c9d8601776269f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fbec7772e9e9acb93c9d8601776269f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c6572c33f47d9bd3ef47b3bdef5bde37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e218a02fb81d4491a5429b99c8c6641(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16384, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_81e92b6774203d74c0434b566b8f498c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_61876070974edcb5b088df143b08f0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(43, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7911037999e5f9d9846dbf02fe637b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_2f4e5c9d90f968a3a1ce3bb082565cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79359eb4197d658b4faa41dfe82264d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
        ]


class TestPrimitiveOp_00e28804eac776c419c249adcdede230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a42bfd090c48fa83bb9538692600a70f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_deb804c583baa286be9e9fc9ea8b54e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9e78cc7662c7e4720ee8d7f3919bd07
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
        ]


class TestPrimitiveOp_7a1d411bf0a2ffdf62fed975df1235e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64d7b1d64ea80364778dc57833d51d28
    def get_inputs(self):
        return [
            paddle.uniform([2092, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8747dd1b2a059467d98ac406f3d596a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8e727f695eb3ec720b984122c008f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2092, 4], dtype='int64'),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7dcb969edcd26a755a9d4a266fdbdf7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0b9fcdb09a831c89edd47d77e85fd3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5c36366e9691196fd50d3c8fec28e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18fa30e4c4319849bb6c7a0db160a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18fa30e4c4319849bb6c7a0db160a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_00c5287fd3344da1304f2d1ca38e4041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_e60cabfeb449d9e6fe5cbf226ab5d43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(3024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_5ea569821d90632c221d87d698d47aad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_94d14616b488d8d566901bf7a89b5cfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd4214b3ac8bece76c586fa7930b5e01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94d14616b488d8d566901bf7a89b5cfd
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[72], dtype='int64'),
        ]


class PrimitiveOp_8cdb30031f64a2a6b8e5b5d3801d1913(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbcea950a1047267704d528308fe0807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cdb30031f64a2a6b8e5b5d3801d1913
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
        ]


class PrimitiveOp_a091d29ce208b18010f0de10076cd762(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96576711f8439389a93c9322892a7a0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a091d29ce208b18010f0de10076cd762
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype='int64').reshape([18]),
        ]


class PrimitiveOp_eb46c588d5c548e47e4c1ebcb155f687(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6804, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4d84b7c3ba4f474de83333f05978884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb46c588d5c548e47e4c1ebcb155f687
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4d84b7c3ba4f474de83333f05978884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb46c588d5c548e47e4c1ebcb155f687
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_654f8f7456f4cf4e7b4e7be537459a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1174, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_b52f19b558995a016f6d0faad12b4764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d754c6e54b0e69d20393fc36cf78f077
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9883fe5e546b0fb651eebb8f6964b485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_4ca37ed42a51c779fbbd29b45c3219a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0b9fcdb09a831c89edd47d77e85fd3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5c36366e9691196fd50d3c8fec28e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_81e92b6774203d74c0434b566b8f498c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18fa30e4c4319849bb6c7a0db160a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2899d3b44250fa214eeca9c41c95530b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_468cb29e0a36fff21e529376dcdd426d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab1ce2d22110e2915ecdb583b83d796c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2284cef299175b997e6d804c0495c2ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c9823ca4ce4f8796c0132f4760b2227d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_cb252d5c06ea9a6003b6fef4eddfa8e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f79359eb4197d658b4faa41dfe82264d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 4], dtype='int32'),
        ]


class TestPrimitiveOp_728edcc7f23d4601ab0c0573c7a3fe5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a42bfd090c48fa83bb9538692600a70f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_3b5fd9176f4ca8f97435145aa44315d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9e78cc7662c7e4720ee8d7f3919bd07
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 68], dtype='int32'),
        ]


class TestPrimitiveOp_234355185aba55524864a604723a6602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64d7b1d64ea80364778dc57833d51d28
    def get_inputs(self):
        return [
            paddle.uniform([4214, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4939fa88583289cb62593c0e9283928c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c8e727f695eb3ec720b984122c008f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4214, 4], dtype='int64'),
        ]


class TestPrimitiveOp_654f8f7456f4cf4e7b4e7be537459a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1174, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_381b7d38045f834cfc3793f6e2055266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f57cf0bd37c3c802a6a37943aa1ae7ef
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7c45dd2f50c02131f5eaed064dea2ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7c45dd2f50c02131f5eaed064dea2ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_7577a7a77dfb86679aa10e810f1de95f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56c08c19ed98775464b2d43786f1079a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1024, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d072c8f2da355d5f47c1e393e5e07496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_d072c8f2da355d5f47c1e393e5e07496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_a76a192a1a376e354c649c29e2a6a604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67a4d78108e57ce95f7ea112bb7b2ac6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(4096, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fbec7772e9e9acb93c9d8601776269f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fbec7772e9e9acb93c9d8601776269f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_c6572c33f47d9bd3ef47b3bdef5bde37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0881eb1cdaa1a60c3858e20c54c9db14
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e218a02fb81d4491a5429b99c8c6641(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(16384, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6a7a1c09dc17e1eff938d95b513b6318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(11, dtype='int32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()