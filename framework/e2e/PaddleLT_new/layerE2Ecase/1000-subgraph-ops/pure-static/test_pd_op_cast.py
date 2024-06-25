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


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_325a54cdbb69996f347272a9b3c99207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class PrimitiveOp_c7f3c07315ad2188af2b84daedc71666(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32, 32, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe32e9ff4a6eb13f8bea9c468d5de67d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7f3c07315ad2188af2b84daedc71666
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


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_922a1b883e0bf4aeed6a9cbd2a2b56db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class PrimitiveOp_3ac872dcea72fc3a635f4048ca1d5814(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 64, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab711b07e2405f3e415e0af0bd1f2a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ac872dcea72fc3a635f4048ca1d5814
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


class PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[128], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_064ef6add347feb6ee549d0214397688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
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


class TestPrimitiveOp_064ef6add347feb6ee549d0214397688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class PrimitiveOp_22bbe85eb172340859cb7cd8253de2ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[128, 128, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b905871e2f11518cc4c4160ae9a0c633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22bbe85eb172340859cb7cd8253de2ed
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


class PrimitiveOp_e3d0cf43f9464ca876f8c3889860417d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 28, 28], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fcf11c0e8736a5f45e7289dba08de9a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d0cf43f9464ca876f8c3889860417d
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


class PrimitiveOp_80dd1577d2ba21d05e76c037cf385c43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1ea3aec9d748f3fafb9f94d0037ec47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80dd1577d2ba21d05e76c037cf385c43
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d1ea3aec9d748f3fafb9f94d0037ec47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80dd1577d2ba21d05e76c037cf385c43
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_cfe18c5693440573076ecc511f4ab7ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1cd0eca299d5fdf59a957fa5a1b72db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfe18c5693440573076ecc511f4ab7ad
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


class PrimitiveOp_2e2276980bad6aec7ea86e78f6504a75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8732, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b3a2f4ff58b24b9ebb20a66c53868dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e2276980bad6aec7ea86e78f6504a75
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_168739ac94130efbf1698bee42da5ead(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_4e3edc49475ff532ecbbac5ab4177dc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5caa92980d1639246e2e291c0b822d20
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


class PrimitiveOp_0bfe0a1ca9e9495647a63b4e675a11ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50a5f36266a860499ed2a1a7d6360de8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bfe0a1ca9e9495647a63b4e675a11ef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45525065064430237, 0.027886996045708656, 0.05469066649675369, 0.09922953695058823, 0.43944913148880005, 0.14764128625392914, 0.2596937119960785, 0.0014553589280694723, 0.2519356906414032, 0.26656362414360046, 0.04113401472568512, 0.14312632381916046, 0.42150378227233887, 0.17673715949058533, 0.0016911025159060955, 0.059259019792079926], dtype='float32').reshape([16]),
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


class TestPrimitiveOp_f1481a3b5f2a16c80862833cc5f7015c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bfbb0d4cd6c81b6f8cfc01ec7a6ea50
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
        ]


class TestPrimitiveOp_dcf5d1de461d618da1188251c84d8c4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bfbb0d4cd6c81b6f8cfc01ec7a6ea50
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


class PrimitiveOp_10d15701845c161c8af2e92a8e5c3365(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 28, 28], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f06f322747066895b4e30f1ec7210b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10d15701845c161c8af2e92a8e5c3365
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


class PrimitiveOp_1d396cc72fc1944924f674fdda50abe8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43de1cc3044190280f1a42cb896a7636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
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


class TestPrimitiveOp_e29404b5e2ffedc9bf1f2cbf8ec65b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
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


class PrimitiveOp_635e2719562390e0d417b33ab5360fe9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e661682815b22d01c35aec758ed4524a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635e2719562390e0d417b33ab5360fe9
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_65e472f51c910f4abc170a023dcbc990(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d66ce875fdedb9bc9b3d66e63bf30f56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65e472f51c910f4abc170a023dcbc990
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
        ]


class PrimitiveOp_de118096f4c00b7740edd95b5ae46aba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6117c3dbefc1b4f93a60c900dcbab694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de118096f4c00b7740edd95b5ae46aba
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_c4ca4be9963981678e05bacd8451f2e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ed082cc3274a4e1e1d777b765be52b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4ca4be9963981678e05bacd8451f2e3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 68], dtype='int32'),
        ]


class PrimitiveOp_72a58bf59689372e6207dfd22d59cd50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_edcfc044c753b56e27f47202e353cea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72a58bf59689372e6207dfd22d59cd50
    def get_inputs(self):
        return [
            paddle.uniform([1787, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d0011eae05562991720812681f4c5f7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_374de2a876f70c4c8cbfa3798defcc4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0011eae05562991720812681f4c5f7a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1787, 4], dtype='int64'),
        ]


class TestPrimitiveOp_c98ce3ab1ca1343149618d5302dc606e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
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


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
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


class TestPrimitiveOp_922a1b883e0bf4aeed6a9cbd2a2b56db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_325a54cdbb69996f347272a9b3c99207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


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


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
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


class PrimitiveOp_b1cd01119c19049abfa5fe3d56771225(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1aa1d38151fc5ca20cbdfc76f5f058cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1cd01119c19049abfa5fe3d56771225
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3d2e0160185b57b046b9fbf67bb58c86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef419b58c7e897df83a0c99ef8d0220d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d2e0160185b57b046b9fbf67bb58c86
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_46aca81dc541bfaf18a7c1a5afd3fe9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f6db2df6309b51cc1dd9f89e2b93d264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46aca81dc541bfaf18a7c1a5afd3fe9d
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_32576a7b1534741c4f479e46b953491a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88428ee7bc09f3e77b678a29d01f0851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32576a7b1534741c4f479e46b953491a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 4], dtype='int32'),
        ]


class PrimitiveOp_740efdca4203ff36869a8e6d04ce1731(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74c6d004709bad8d08eb80b48c5c452a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740efdca4203ff36869a8e6d04ce1731
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_0fc3e75a3abf836cfbe0b7f4ee52ae94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a432eabd2777b3f64c03af3040faaba5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fc3e75a3abf836cfbe0b7f4ee52ae94
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 68], dtype='int32'),
        ]


class PrimitiveOp_8268772fa845f366fa666d3e2cfc5a17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58656b88be0cb1b977dec23aad2eeb00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8268772fa845f366fa666d3e2cfc5a17
    def get_inputs(self):
        return [
            paddle.uniform([5585, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0886d5e8e96ff70a3c63f48920c7d358(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc6ad6e4d1bf348537c5b6825135d7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0886d5e8e96ff70a3c63f48920c7d358
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5585, 4], dtype='int64'),
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


class PrimitiveOp_34af7e4c2a6f206b6e81c96ab6c6e196(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a6e814d181d75ed2a1743b4e524c2f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34af7e4c2a6f206b6e81c96ab6c6e196
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a7849b832fe73200366b2cfeb97f2693(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
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


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class PrimitiveOp_57b68ee16ee8d368351b8c7c2a94f8b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47c54a219330a6dde5a48b6691743740(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57b68ee16ee8d368351b8c7c2a94f8b0
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_bbcea950a1047267704d528308fe0807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cdb30031f64a2a6b8e5b5d3801d1913
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


class PrimitiveOp_30407e04dde1abc0b116b52e6babccc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_def88ac01330edaa673bc57308b853b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30407e04dde1abc0b116b52e6babccc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_de562efec81f78f5dbec1af75c650af8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e2975575703795392fc93df970f61c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de562efec81f78f5dbec1af75c650af8
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


class TestPrimitiveOp_daf75dea7e93a1bd7c05ecab514f4d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5caa92980d1639246e2e291c0b822d20
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
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


class TestPrimitiveOp_1aa1d38151fc5ca20cbdfc76f5f058cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1cd01119c19049abfa5fe3d56771225
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef419b58c7e897df83a0c99ef8d0220d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d2e0160185b57b046b9fbf67bb58c86
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


class PrimitiveOp_9478d280232355ab9902d47d54749ccb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a78cfcf715aff67351ef2b098fb1aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9478d280232355ab9902d47d54749ccb
    def get_inputs(self):
        return [
            paddle.to_tensor([False, False, False, True, True, False], dtype='bool').reshape([6]),
        ]


class TestPrimitiveOp_85fe73d4c37362aed672db2c43515ffb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9478d280232355ab9902d47d54749ccb
    def get_inputs(self):
        return [
            paddle.to_tensor([False, False, False, False, True, False], dtype='bool').reshape([6]),
        ]


class TestPrimitiveOp_e661682815b22d01c35aec758ed4524a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635e2719562390e0d417b33ab5360fe9
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d66ce875fdedb9bc9b3d66e63bf30f56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65e472f51c910f4abc170a023dcbc990
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
        ]


class TestPrimitiveOp_6117c3dbefc1b4f93a60c900dcbab694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de118096f4c00b7740edd95b5ae46aba
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_5d91d46da38a6aafdb1fb71922d92f0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 76], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17bed0f4b53b75e092f46a63c4ebe5a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d91d46da38a6aafdb1fb71922d92f0c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 76], dtype='int32'),
        ]


class PrimitiveOp_cdd913b1ca0fa7b11cea78045531e432(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40c216d6376fdd69919ba22de8f03817(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd913b1ca0fa7b11cea78045531e432
    def get_inputs(self):
        return [
            paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_62700a8d197a33ba6235d8e3db5ced59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8065b0d3e89a3ca4cc145c1cce11873a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62700a8d197a33ba6235d8e3db5ced59
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1774, 4], dtype='int64'),
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


class PrimitiveOp_d34f9eac6599e1e2c2a9181b0d5c2f55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fd8251d63401ae22276328ab00dcb16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d34f9eac6599e1e2c2a9181b0d5c2f55
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a8c48ab8a91874df17a5c53f5656df6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbfb7085a1453375a8a50e43bf679e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8c48ab8a91874df17a5c53f5656df6c
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


class PrimitiveOp_31c83153cd11d968a37becea3e407770(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 256, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_68840ae1fff0e6eabfd021b0c77b69c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c83153cd11d968a37becea3e407770
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbfb7085a1453375a8a50e43bf679e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8c48ab8a91874df17a5c53f5656df6c
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


class TestPrimitiveOp_0025bc7ba8dfe04cabaa1dfe69b9ff39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
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


class PrimitiveOp_b4a387e00a223fbc321d6bdc768811ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82c1e2962d27218163b5839e4076f01b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4a387e00a223fbc321d6bdc768811ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bbbb11b99efb50395ec4043d7a8d6ebb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b261dcf7eb61190729852a3da6f10a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbbb11b99efb50395ec4043d7a8d6ebb
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


class PrimitiveOp_fbbf721f9dd90e4d18449733353f118c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8191adc454c438cf45e8b27f2950cfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbbf721f9dd90e4d18449733353f118c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4237799346446991, 0.11541059613227844, 0.00809119176119566, 0.3815240263938904, 0.059733111411333084, 0.08381999284029007, 0.3550366163253784, 0.2367265820503235, 0.3807695508003235, 0.32798030972480774, 0.31185442209243774, 0.3005569875240326, 0.4866105914115906, 0.01647285930812359, 0.10545752942562103, 0.422288715839386, 0.1458156257867813, 0.02166130021214485, 0.21083933115005493, 0.0744839757680893, 0.0265908632427454, 0.24041879177093506, 0.38260388374328613, 0.10905811190605164], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_7ea8f49889676bd664c8c948b148cd8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b96b6cfd491ffd9960c841493858128d
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
        ]


class TestPrimitiveOp_f06436bb8f4f2d83f7b3911c545af1b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b96b6cfd491ffd9960c841493858128d
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


class TestPrimitiveOp_01d7bc53956ec052456b8c992979f4d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa66f160a3e34528e0d691b1ea16f430
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31928032636642456], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2c5ec52f83318fd73f8f3ab0e19d57d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_325a54cdbb69996f347272a9b3c99207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
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


class TestPrimitiveOp_325a54cdbb69996f347272a9b3c99207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_fe32e9ff4a6eb13f8bea9c468d5de67d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7f3c07315ad2188af2b84daedc71666
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


class TestPrimitiveOp_922a1b883e0bf4aeed6a9cbd2a2b56db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
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


class TestPrimitiveOp_922a1b883e0bf4aeed6a9cbd2a2b56db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_ab711b07e2405f3e415e0af0bd1f2a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ac872dcea72fc3a635f4048ca1d5814
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


class TestPrimitiveOp_064ef6add347feb6ee549d0214397688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
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


class TestPrimitiveOp_064ef6add347feb6ee549d0214397688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_b905871e2f11518cc4c4160ae9a0c633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22bbe85eb172340859cb7cd8253de2ed
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


class PrimitiveOp_ed4b92cc994dae57ccc5ef21cd854d95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59381ef6b602138d9d7236d1d880a1f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed4b92cc994dae57ccc5ef21cd854d95
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_8c040199f6fbda31c2782e1c48c87306(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8aed368ab35b895c2b0d8cb2ee8532e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c040199f6fbda31c2782e1c48c87306
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 4], dtype='int32'),
        ]


class PrimitiveOp_06d67ea732b58de1bd5e29078495b62d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_088711616939697256fc4d2840e9d7c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06d67ea732b58de1bd5e29078495b62d
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_ae6cb3d616044ff184f0c31db184b0f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01df8bdbae71b5dc55b191ae54fc1692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae6cb3d616044ff184f0c31db184b0f6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 68], dtype='int32'),
        ]


class PrimitiveOp_1ca005e2ce0a9b99ea934a1a3fe6728c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6733952ed24fe37c1eaf72e0ce909b3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ca005e2ce0a9b99ea934a1a3fe6728c
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_33cdca0e1c509b499785baf6d43e07d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_741268257c7e8c57d72763c213365c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33cdca0e1c509b499785baf6d43e07d4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1501, 4], dtype='int64'),
        ]


class TestPrimitiveOp_e661682815b22d01c35aec758ed4524a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635e2719562390e0d417b33ab5360fe9
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_e661682815b22d01c35aec758ed4524a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635e2719562390e0d417b33ab5360fe9
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_77df59e6886e0065628e1abe64c00b38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b26a14f6a342a6c8a8b15e77534068d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77df59e6886e0065628e1abe64c00b38
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


class PrimitiveOp_503a0a713d33c2562f6d124c48f21506(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_750ac7a81ea1a095c159488e9faeda44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_503a0a713d33c2562f6d124c48f21506
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43235915899276733, 0.37341681122779846, 0.39400672912597656, 0.1910826563835144], dtype='float32').reshape([4]),
        ]


class PrimitiveOp_a332621205b828c0d4f79b40a53ad5fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db0d057bf3de8df5813f561a853da3d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a332621205b828c0d4f79b40a53ad5fc
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6d11059e78a4a495e65a37fe1a1fb772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a332621205b828c0d4f79b40a53ad5fc
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


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
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


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
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


class PrimitiveOp_cd6aaeb750a50459ec34ce64f49b4f1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b4eb72551b5955733884a929b816e76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd6aaeb750a50459ec34ce64f49b4f1f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7b4eb72551b5955733884a929b816e76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd6aaeb750a50459ec34ce64f49b4f1f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_147a9b6f6fe29cb6485756fdbd24aae0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ff40e5649f1695942c3c36c3f97b678(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_147a9b6f6fe29cb6485756fdbd24aae0
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


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
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


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
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


class TestPrimitiveOp_27808353ddcc94f4a5023e2499eb03c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0025bc7ba8dfe04cabaa1dfe69b9ff39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7b4eb72551b5955733884a929b816e76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd6aaeb750a50459ec34ce64f49b4f1f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_33988b180b9c91806a7a63416c23173d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9a65fcb912afed47f83ac35e328577ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33988b180b9c91806a7a63416c23173d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
        ]


class PrimitiveOp_f797db38af0652bbfba2f8521b072451(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1415c717465b40aed3886713d522bb96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f797db38af0652bbfba2f8521b072451
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_8d26a657649ea0594e9bdf80b6fbeb27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_008deaedf2880293540490fb93d39c14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d26a657649ea0594e9bdf80b6fbeb27
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
        ]


class PrimitiveOp_a0d50f98edc83a7d7fd442047b31835c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1fba4bf521030f9342d3e0f162bfc447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0d50f98edc83a7d7fd442047b31835c
    def get_inputs(self):
        return [
            paddle.uniform([2049, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3b3757f6c382964e2c21ece0d918d4f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5df6968081290db237d8aadf0f5adf5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b3757f6c382964e2c21ece0d918d4f0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2049, 4], dtype='int64'),
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


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
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


class PrimitiveOp_d5438f81faff01b739697c21922cbc54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_961dee125a6f99aabffe2df18a85d08e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5438f81faff01b739697c21922cbc54
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_27593974db80c06957d17e1b912923de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58d4c6338f6584f1a82e8996be9dda5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27593974db80c06957d17e1b912923de
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 4], dtype='int32'),
        ]


class PrimitiveOp_9162f58d5b2be0e62a895ceee831477e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e82dd9265048d2b5dc5c72de4b193263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9162f58d5b2be0e62a895ceee831477e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_da7006d565f58cd99f001a5741c05d23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70e18b8ee24ede104cdf3abf6eae87f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da7006d565f58cd99f001a5741c05d23
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 68], dtype='int32'),
        ]


class PrimitiveOp_e1fdaa63dfba0c4b538e95282998d98f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56b2d9d93e3adb6cdf2693ebfddf8637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1fdaa63dfba0c4b538e95282998d98f
    def get_inputs(self):
        return [
            paddle.uniform([4634, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f25034e8643e02d7f399045639f945ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1710ac86fc61b0940accbefc77c138da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f25034e8643e02d7f399045639f945ae
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4634, 4], dtype='int64'),
        ]


class PrimitiveOp_992969dd9ae65208274b1209a2cb6b55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 28, 28], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aeb47201fbeedcdeb7126ee42052a55b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992969dd9ae65208274b1209a2cb6b55
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[6, 28, 28], dtype='int32'),
        ]


class PrimitiveOp_d6b778b8fa3f3641fe49b1fac39fe2e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2434, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24688bbc99e321d02a60140428becf10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6b778b8fa3f3641fe49b1fac39fe2e7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d1ea3aec9d748f3fafb9f94d0037ec47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80dd1577d2ba21d05e76c037cf385c43
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_3936d8f47391a74848a6f8c60717cc42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19abfdc4bc2a0d3f9274901058f1c5ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3936d8f47391a74848a6f8c60717cc42
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 4], dtype='int32'),
        ]


class PrimitiveOp_147782535a6c6234184a8ef44aaeb5cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_034723bfa1b05ed2a8dc962b8f15cd06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_147782535a6c6234184a8ef44aaeb5cb
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_7a46795b6cab6ee0090890ee02b933eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c7e2c8df89ae1771e532ae0bc4ce637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a46795b6cab6ee0090890ee02b933eb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 68], dtype='int32'),
        ]


class PrimitiveOp_daf43d38da30c27cd05be7160ff04a48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be61b5307d4edd940049e7aaf260d4f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_daf43d38da30c27cd05be7160ff04a48
    def get_inputs(self):
        return [
            paddle.uniform([1000, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_536eda2b097ecf1bd7f224c725fe4b7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c90b8d2ca5b90f59a043a580428f82bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_536eda2b097ecf1bd7f224c725fe4b7e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1000, 4], dtype='int64'),
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


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
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


class TestPrimitiveOp_7a6e814d181d75ed2a1743b4e524c2f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34af7e4c2a6f206b6e81c96ab6c6e196
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
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


class TestPrimitiveOp_7a6e814d181d75ed2a1743b4e524c2f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34af7e4c2a6f206b6e81c96ab6c6e196
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
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


class TestPrimitiveOp_7a6e814d181d75ed2a1743b4e524c2f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34af7e4c2a6f206b6e81c96ab6c6e196
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
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


class PrimitiveOp_523ecb4e70a73deee1b3afde7d7d35c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0aebf65fa2afd0a311b4a6707ae20b2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_523ecb4e70a73deee1b3afde7d7d35c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1ba589dc31bede712445f5f0914341d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 1, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_330ab7ad6b1ec490a0382ec0749bc444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1ba589dc31bede712445f5f0914341d
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


class TestPrimitiveOp_064ef6add347feb6ee549d0214397688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
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


class TestPrimitiveOp_064ef6add347feb6ee549d0214397688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_b905871e2f11518cc4c4160ae9a0c633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22bbe85eb172340859cb7cd8253de2ed
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


class TestPrimitiveOp_922a1b883e0bf4aeed6a9cbd2a2b56db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
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


class TestPrimitiveOp_922a1b883e0bf4aeed6a9cbd2a2b56db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_ab711b07e2405f3e415e0af0bd1f2a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ac872dcea72fc3a635f4048ca1d5814
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


class TestPrimitiveOp_325a54cdbb69996f347272a9b3c99207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
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


class TestPrimitiveOp_325a54cdbb69996f347272a9b3c99207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_fe32e9ff4a6eb13f8bea9c468d5de67d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7f3c07315ad2188af2b84daedc71666
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


class TestPrimitiveOp_66cb74165486cb8db056065a8784ff2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bfbb0d4cd6c81b6f8cfc01ec7a6ea50
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


class TestPrimitiveOp_66cb74165486cb8db056065a8784ff2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bfbb0d4cd6c81b6f8cfc01ec7a6ea50
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
        ]


class PrimitiveOp_dcf423e8a8c00152d2a4d3f2a3d944c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 16, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a3033d0430bb680410085b15afd70e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcf423e8a8c00152d2a4d3f2a3d944c3
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


class PrimitiveOp_5beaef702f0af29738f36df1b2675908(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1c3701b90e8a7f8d9d783e533ff733e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5beaef702f0af29738f36df1b2675908
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


class TestPrimitiveOp_c1c3701b90e8a7f8d9d783e533ff733e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5beaef702f0af29738f36df1b2675908
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
        ]


class PrimitiveOp_7f710d4879a0b36d6f2c72b81d1b385a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 8, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af6f2c6b1d99e6271a1aac5bbd592b1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f710d4879a0b36d6f2c72b81d1b385a
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


class PrimitiveOp_0a406bfd330bcdfa79c201703642b76f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d02aa11c610588b258328617e24a0f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a406bfd330bcdfa79c201703642b76f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
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


class TestPrimitiveOp_d02aa11c610588b258328617e24a0f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a406bfd330bcdfa79c201703642b76f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
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


class TestPrimitiveOp_d02aa11c610588b258328617e24a0f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a406bfd330bcdfa79c201703642b76f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
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


class PrimitiveOp_7ce3402eb1e1a78703abfa6a22d1ed86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3de942a474b862849e626634ab88d925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3402eb1e1a78703abfa6a22d1ed86
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_330ab7ad6b1ec490a0382ec0749bc444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1ba589dc31bede712445f5f0914341d
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


class PrimitiveOp_acad1b6d15f9c4153bda3d8a99f420d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_acf270fc5529c009271adc2a4fa2a02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acad1b6d15f9c4153bda3d8a99f420d4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_a08b14606d789ec62bb8d8edbc17789b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f3e6d93f1805494a61e0d05575eafd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a08b14606d789ec62bb8d8edbc17789b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 4], dtype='int32'),
        ]


class PrimitiveOp_661bdedd9362524196ae517f04ae6441(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_982609a26ae840acce7745e39ec37c42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_661bdedd9362524196ae517f04ae6441
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_4bcadaee065e0ad0a14e20bdbdc12f6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a3b41476016e490e46c713a69d89559(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bcadaee065e0ad0a14e20bdbdc12f6d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 68], dtype='int32'),
        ]


class PrimitiveOp_ee0a281862e05bdeca26c86f91017946(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a93abd6b8a2575f676e9b7b3ea7a8069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee0a281862e05bdeca26c86f91017946
    def get_inputs(self):
        return [
            paddle.uniform([2382, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e448338fa09aba12e7fc6f36162cade0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_453e933bbf13f268dfad342c1d2e6303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e448338fa09aba12e7fc6f36162cade0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2382, 4], dtype='int64'),
        ]


class TestPrimitiveOp_b0da880982d26ce7700fc842dab18f8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(22, dtype='int32').reshape([]),
        ]


class PrimitiveOp_014953e228d3107afdebf529cab6c848(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e29b2fa7df34a607d2adf78ac9472e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_014953e228d3107afdebf529cab6c848
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_b882e21cf0f33be86e6ad9ce968ff51a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c99f4c282e77d55cc42a0deb8415328f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b882e21cf0f33be86e6ad9ce968ff51a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 4], dtype='int32'),
        ]


class PrimitiveOp_21c9131cdbee3b081705db7e8650e0e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a692182b8705c36fa2cdd7fe941f76e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21c9131cdbee3b081705db7e8650e0e1
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_eb2885c59641af627e93a1313280ee5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13657966194f845579d47390a5f2fa1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb2885c59641af627e93a1313280ee5d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 68], dtype='int32'),
        ]


class PrimitiveOp_e7f2bf29ec9dc30b8ac8324a09a42e9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a37674f5597d77c682ab2b565118a55d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7f2bf29ec9dc30b8ac8324a09a42e9c
    def get_inputs(self):
        return [
            paddle.uniform([2976, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0b6b642fcc4b64629f3e26a9ba245905(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8ae2c546cf8db5dfaff2ccdea0ca4ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b6b642fcc4b64629f3e26a9ba245905
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2976, 4], dtype='int64'),
        ]


class PrimitiveOp_d2370c392870d67783e60fe9c0d2ef1a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e34bfc7284f58e256189cd5a168872dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2370c392870d67783e60fe9c0d2ef1a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_e7cdcfe8eeb455f70178643375d5c4c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_edcc41ac241cc1e6cbf3f4bde5c25050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7cdcfe8eeb455f70178643375d5c4c9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 4], dtype='int32'),
        ]


class PrimitiveOp_b5be88edbe24d7a56618e848d479ffbe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1279310c29f0e1d0fcd5b014e63eb18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5be88edbe24d7a56618e848d479ffbe
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_42e11b0251239d654f34a8401a2665a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3762f65c16c188199a5d5d826e92fe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e11b0251239d654f34a8401a2665a6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 68], dtype='int32'),
        ]


class PrimitiveOp_664f411c333ecb2f4590ea6f37fdf3de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47fa71de176757b63898c352b809e346(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_664f411c333ecb2f4590ea6f37fdf3de
    def get_inputs(self):
        return [
            paddle.uniform([3753, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3f312183ce324f1816e268f3b2f8f0ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7cda1643236a1278ddf36e378ce8afe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f312183ce324f1816e268f3b2f8f0ed
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3753, 4], dtype='int64'),
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


class TestPrimitiveOp_dd5f211485d6277c4da4778daee93036(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
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


class PrimitiveOp_18ecd9fe74523bff58464e616028ce7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 28, 28], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32a51674f53f44504bbaeb4278ecba26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18ecd9fe74523bff58464e616028ce7f
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


class PrimitiveOp_623d28febb5adf1faa5405d3d56324f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31ebee961c5bacbf504d4a6ae7507e46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_623d28febb5adf1faa5405d3d56324f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b261dcf7eb61190729852a3da6f10a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbbb11b99efb50395ec4043d7a8d6ebb
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


class PrimitiveOp_b0c435ab25f1af43b55b7413d1671866(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75006c8f0d349181cf41d7f6d33b57e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0c435ab25f1af43b55b7413d1671866
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbfb7085a1453375a8a50e43bf679e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8c48ab8a91874df17a5c53f5656df6c
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


class PrimitiveOp_74803e472818df185c306091156cbd95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7cfe4e8680b42d0b8dedc03000473698(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74803e472818df185c306091156cbd95
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4680095911026001, 0.41818323731422424, 0.14651672542095184, 0.280924916267395, 0.4328104853630066, 0.054005738347768784, 0.3026198148727417, 0.2460148185491562, 0.2609083652496338, 0.20063157379627228, 0.007730483077466488, 0.14513055980205536, 0.1574631929397583, 0.22134825587272644, 0.48197489976882935, 0.39677584171295166, 0.10996971279382706, 0.21447205543518066, 0.26940596103668213, 0.01980494149029255], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_7914332f4b5eb1edfcd56550f9c7090d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b21bfbfb24128ef45526e251e7c4e58
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
        ]


class TestPrimitiveOp_d724c72e68cc04d1751fc0096853e057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b21bfbfb24128ef45526e251e7c4e58
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


class TestPrimitiveOp_325a54cdbb69996f347272a9b3c99207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
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


class TestPrimitiveOp_325a54cdbb69996f347272a9b3c99207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_fe32e9ff4a6eb13f8bea9c468d5de67d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7f3c07315ad2188af2b84daedc71666
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


class TestPrimitiveOp_922a1b883e0bf4aeed6a9cbd2a2b56db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
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


class TestPrimitiveOp_922a1b883e0bf4aeed6a9cbd2a2b56db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_ab711b07e2405f3e415e0af0bd1f2a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ac872dcea72fc3a635f4048ca1d5814
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


class TestPrimitiveOp_064ef6add347feb6ee549d0214397688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
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


class TestPrimitiveOp_064ef6add347feb6ee549d0214397688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_b905871e2f11518cc4c4160ae9a0c633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22bbe85eb172340859cb7cd8253de2ed
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


class TestPrimitiveOp_27808353ddcc94f4a5023e2499eb03c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
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


class TestPrimitiveOp_7b4eb72551b5955733884a929b816e76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd6aaeb750a50459ec34ce64f49b4f1f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9a65fcb912afed47f83ac35e328577ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33988b180b9c91806a7a63416c23173d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
        ]


class TestPrimitiveOp_1415c717465b40aed3886713d522bb96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f797db38af0652bbfba2f8521b072451
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_008deaedf2880293540490fb93d39c14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d26a657649ea0594e9bdf80b6fbeb27
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
        ]


class PrimitiveOp_afc4b2adc13b1c60b7bc53da31aa2ceb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49f5dda746d7e80d78c0f4c6d9d00acb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afc4b2adc13b1c60b7bc53da31aa2ceb
    def get_inputs(self):
        return [
            paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5df710ba95a108391221759977d99757(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f69f5f7b2439dc7932f2ab5f76940c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5df710ba95a108391221759977d99757
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1995, 4], dtype='int64'),
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


class PrimitiveOp_55b94a50b5ef81c536f28f6d618615fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7cd49abbb36f9a535f98fb5cd101b742(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55b94a50b5ef81c536f28f6d618615fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
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


class TestPrimitiveOp_43de1cc3044190280f1a42cb896a7636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
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


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
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


class TestPrimitiveOp_ab636bee62813cbd19f34501696fc38e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5caa92980d1639246e2e291c0b822d20
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_5e8e56454554e6011455fec57b1e522b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
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


class TestPrimitiveOp_d02aa11c610588b258328617e24a0f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a406bfd330bcdfa79c201703642b76f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_27808353ddcc94f4a5023e2499eb03c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
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


class TestPrimitiveOp_82c1e2962d27218163b5839e4076f01b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4a387e00a223fbc321d6bdc768811ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b261dcf7eb61190729852a3da6f10a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbbb11b99efb50395ec4043d7a8d6ebb
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


class PrimitiveOp_13ad6d900d5aff5d8b6b9d48b868af01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_276c3ad51e3ef26b4dc0f69d90083e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ad6d900d5aff5d8b6b9d48b868af01
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_1e337b65abc56922f7a44586ba9f5a50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7758965d23cc5c7d472b4bf07300bdbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e337b65abc56922f7a44586ba9f5a50
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 4], dtype='int32'),
        ]


class PrimitiveOp_2b006d447fdeccf4598d055e342ec949(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de20a575725cd7d49e671f7d7cc81c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b006d447fdeccf4598d055e342ec949
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_4fb60febf02b154e98728c1e82f69459(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f0831acdd76c2745038b9d185a59f58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fb60febf02b154e98728c1e82f69459
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 68], dtype='int32'),
        ]


class PrimitiveOp_47073979e731d4b1633b24d4f4d7652f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c5df14c59da355526c837dcbb9981f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47073979e731d4b1633b24d4f4d7652f
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_61e345c087a9b678ccc71583a223bf53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_71a1fea7704888439f0ba976557eb46f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e345c087a9b678ccc71583a223bf53
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4185, 4], dtype='int64'),
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


class TestPrimitiveOp_378442e6c8c1d1c22188ae416accaf77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
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


class TestPrimitiveOp_325a54cdbb69996f347272a9b3c99207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
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


class TestPrimitiveOp_325a54cdbb69996f347272a9b3c99207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_fe32e9ff4a6eb13f8bea9c468d5de67d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7f3c07315ad2188af2b84daedc71666
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


class TestPrimitiveOp_922a1b883e0bf4aeed6a9cbd2a2b56db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
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


class TestPrimitiveOp_922a1b883e0bf4aeed6a9cbd2a2b56db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_ab711b07e2405f3e415e0af0bd1f2a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ac872dcea72fc3a635f4048ca1d5814
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


class TestPrimitiveOp_064ef6add347feb6ee549d0214397688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
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


class TestPrimitiveOp_064ef6add347feb6ee549d0214397688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_b905871e2f11518cc4c4160ae9a0c633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22bbe85eb172340859cb7cd8253de2ed
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