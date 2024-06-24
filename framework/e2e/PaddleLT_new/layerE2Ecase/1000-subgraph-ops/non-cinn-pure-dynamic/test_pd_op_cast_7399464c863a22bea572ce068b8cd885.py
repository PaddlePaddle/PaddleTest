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


class PrimitiveOp_67329cbfbfe6e2e657262692edbd53a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ee71a801e379743fcdf0fe71f22ccf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67329cbfbfe6e2e657262692edbd53a3
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


class PrimitiveOp_74168d360d4449d2f23e4a840a6867b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0675b4c542a99a44b8d3d7ae01cac5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_f14434d7a9c4bb288f3dfac5dbd4ba5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_50f90a45fde59a080114015d64ba3c63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_e68e21bb2938714cb20dad4be8e4a2f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67329cbfbfe6e2e657262692edbd53a3
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


class PrimitiveOp_cb585bbf5f8008f78da86a0c4776f140(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc0db21aae9270b5aa8966b37a532bce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb585bbf5f8008f78da86a0c4776f140
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
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


class TestPrimitiveOp_3e489e6297b43b3dc1a7d03677ffbef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d754c6e54b0e69d20393fc36cf78f077
    def get_inputs(self):
        return [
            paddle.to_tensor([128], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_16fd36231e14aef0eeb96f2613cf7139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d754c6e54b0e69d20393fc36cf78f077
    def get_inputs(self):
        return [
            paddle.to_tensor([16], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b52f19b558995a016f6d0faad12b4764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d754c6e54b0e69d20393fc36cf78f077
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_5ce2dc79f4895e29a99bfdd0bb57c2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[96], dtype='int64'),
        ]


class TestPrimitiveOp_550cd53687d2fc30d40debe837a8af8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[48], dtype='int64'),
        ]


class TestPrimitiveOp_b2d7fc7cbb8d914733373119db97ba25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype='int64').reshape([24]),
        ]


class PrimitiveOp_2cceba588efb2cce6310a59179db9450(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20247ce337eaa5194559517cf8a8d8f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20247ce337eaa5194559517cf8a8d8f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
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


class TestPrimitiveOp_8fd6a3ecc6cb37670331a8ae4c98e926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08915137499570847, 0.0019794581457972527, 0.11768212914466858, 0.4924314320087433, 0.19912375509738922, 0.1807744801044464, 0.04182153195142746, 0.18964476883411407, 0.3268153667449951, 0.22180624306201935, 0.10589847713708878, 0.03320932760834694, 0.18596151471138, 0.2120317816734314, 0.033170074224472046, 0.3353402018547058], dtype='float32').reshape([16]),
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


class PrimitiveOp_c785908ecb825189e48322e59b25894b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b2192006040b2ffe86999e100e14da9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
        ]


class PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3091f2efa85af2ed7062532b7cf4f8c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9284ebb530cba5d4d4d68ed2eb7f4955(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 68], dtype='int32'),
        ]


class PrimitiveOp_0bf33482bd49195c5fe8c0182d505424(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48c975c84e7c5a2a9f8013e28c32e0be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_932e6e6b886037fb5a9978fd771da4cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1745, 4], dtype='int64'),
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


class TestPrimitiveOp_d072c8f2da355d5f47c1e393e5e07496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_7c45dd2f50c02131f5eaed064dea2ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_56d5378017754b901a36c34d3a89ed40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
        ]


class TestPrimitiveOp_25a8ae870ecbaf3cfb96836a1d1ac239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25a8ae870ecbaf3cfb96836a1d1ac239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
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


class PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be1b842380ad18e922bf36eaaccde450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_b49b46e2a07e2a129892ccb6d5198cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 4], dtype='int32'),
        ]


class TestPrimitiveOp_8e0d3b4c356e967e97e82447e93ace3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_5591c2fa5dc6da0222cc1bddd58dd2b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109, 68], dtype='int32'),
        ]


class TestPrimitiveOp_95bcc118bdd673207ec0281c5470c188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d670a32a164e5a4dc94b688e3c3032e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5556, 4], dtype='int64'),
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


class TestPrimitiveOp_cac252f25f8df5fd254d9711d00a191c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_02fd5e6a3b81ae110e26b4be18656c8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6d54d5babe998459821b5febccfe4c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_be1b842380ad18e922bf36eaaccde450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_6449fbdb7e3c9b0e7523dfc28dfa16c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8cd3eac4b0759bcf033ac4911afa05c
    def get_inputs(self):
        return [
            paddle.to_tensor([False, True, True, False, False, True], dtype='bool').reshape([6]),
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


class TestPrimitiveOp_9b2192006040b2ffe86999e100e14da9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
        ]


class TestPrimitiveOp_3091f2efa85af2ed7062532b7cf4f8c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_307fe4c9dd496a2dab7db2e902e95e04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549, 76], dtype='int32'),
        ]


class TestPrimitiveOp_4a29f2cd93626c09ad0fecff0fca2e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54ed4a72b1a5bf62a6249c7116674099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1744, 4], dtype='int64'),
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


class TestPrimitiveOp_3d5607480de507d86aa419547a58bdd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f16aeb04493caaa635761e0dc5bd152b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_1af70197c5b6faca2d441518f495df61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f16aeb04493caaa635761e0dc5bd152b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_8d1c53dc389badbafed1347ab6297f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_5eb3152e5a745b70b56d52cfaf5623d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[80], dtype='int64'),
        ]


class TestPrimitiveOp_3b907a34313df64cd6c82f100723bb3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
        ]


class TestPrimitiveOp_4ada8ff78142a496c452c96a035ae227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype='int64').reshape([20]),
        ]


class TestPrimitiveOp_fd1a0eedf2dd912424c3d384d631080b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd1a0eedf2dd912424c3d384d631080b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6f2994399824be44229a2153e9089cc9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d18030076aadedcfc4c9e2a9ddbb5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f2994399824be44229a2153e9089cc9
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


class TestPrimitiveOp_be854763ca0b5d19ea7d6e2fb56cbad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05784948170185089, 0.43556898832321167, 0.4312872588634491, 0.4016205370426178, 0.46198728680610657, 0.0058740247040987015, 0.16386902332305908, 0.3483448922634125, 0.3245091736316681, 0.36026546359062195, 0.07979327440261841, 0.02710764855146408, 0.06300970911979675, 0.109025739133358, 0.2896069288253784, 0.13761979341506958, 0.318467378616333, 0.38401126861572266, 0.43845245242118835, 0.16789230704307556, 0.10596621036529541, 0.3684060275554657, 0.28659170866012573, 0.26515650749206543], dtype='float32').reshape([24]),
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


class PrimitiveOp_591250b83c1a2579ee9a4464305bddca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7aaf2662c1d5033d067efe2d53943ffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_591250b83c1a2579ee9a4464305bddca
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1601026952266693], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_0675b4c542a99a44b8d3d7ae01cac5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_f14434d7a9c4bb288f3dfac5dbd4ba5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_50f90a45fde59a080114015d64ba3c63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_c8cc6b63a15c6948995add2c335c3fc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 4], dtype='int32'),
        ]


class TestPrimitiveOp_8e18738ec19410f32fcaf1251c7b5886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_4fa8bd3dcf5a7d7dca408da01cdb1bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024, 68], dtype='int32'),
        ]


class TestPrimitiveOp_ac577898cc5d38e49391adab732fa771(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eff85a39d9d69fced1ea915d20fd96c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1547, 4], dtype='int64'),
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


class TestPrimitiveOp_1a898c30a35db5905584b37ea6677735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb585bbf5f8008f78da86a0c4776f140
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
        ]


class TestPrimitiveOp_be2e397a1b0a7ff70440110e54a8e5c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f2994399824be44229a2153e9089cc9
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


class TestPrimitiveOp_938e4c069ccad139957a6c7336f36d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3001556098461151, 0.17418614029884338, 0.20308345556259155, 0.2840249538421631], dtype='float32').reshape([4]),
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


class TestPrimitiveOp_91a6357a931011d7245c3f8a855d6563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype='int64').reshape([14]),
        ]


class TestPrimitiveOp_374ec45affad91f3656a2bc9ed735e9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([14, 14, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63d5976c223c4a012d7da04ae964b458(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([14, 14, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13a66480c01d673fe0216b7fbfad03b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype='int64').reshape([28]),
        ]


class TestPrimitiveOp_46134c0535804825d30982e7393a1dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([28, 28, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_332bbe81474be517e2bd86a236cafffd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([28, 28, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad2bdca6ebeab0590833a19a60e4578f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[56], dtype='int64'),
        ]


class TestPrimitiveOp_40a43d0c4f09222d9c36f31e6890f554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([56, 56, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f414c9cd35fcb8d453e4493626b6865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_91c606508ed1324fd1e821807330321a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb585bbf5f8008f78da86a0c4776f140
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


class TestPrimitiveOp_9ee71a801e379743fcdf0fe71f22ccf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67329cbfbfe6e2e657262692edbd53a3
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


class TestPrimitiveOp_d8a6ab8c2d3f75e4733c332dda561fe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
        ]


class TestPrimitiveOp_145a6d15ba2fcc9ee0ff11e2c73bc351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9e63f55bcbcd7c88aaf555400a977e66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
        ]


class TestPrimitiveOp_d6006791e66193d33a7c2a5ec1b96589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e86e527598b1651d1ea9a6b921de144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2056, 4], dtype='int64'),
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


class TestPrimitiveOp_631e4f4d7cce651ee0e7835aad5c698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 4], dtype='int32'),
        ]


class TestPrimitiveOp_49ed7fd1dd416a2556d890435285925c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f439c43fc3fcd05ff3179ecba3aecbda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261, 68], dtype='int32'),
        ]


class TestPrimitiveOp_9ac950f4c4d4287bbb069477e923226a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3d5bd4fd2fdcc9ff98bc247ec3f4147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4650, 4], dtype='int64'),
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


class TestPrimitiveOp_f70997e677a3cacd64014f88117f05f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 4], dtype='int32'),
        ]


class TestPrimitiveOp_1dc3a633558b3f9fca23d07a0f403dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_addefef4cc51748300bb1e7737745a2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100, 68], dtype='int32'),
        ]


class TestPrimitiveOp_5dde604fba183e05e0aa92227086f8a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf4b56dd73be2b966cba3b1e58e5caf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1059, 4], dtype='int64'),
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


class TestPrimitiveOp_a1e456ba27d74fc52221cc698987d47d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[68], dtype='int64'),
        ]


class TestPrimitiveOp_07182578b12b31c764b0b175e2e5656c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[34], dtype='int64'),
        ]


class TestPrimitiveOp_a46bb26a15c7f9a911730ba19b3f294d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype='int64').reshape([17]),
        ]


class TestPrimitiveOp_6d169da80271883012406add76ea0d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d169da80271883012406add76ea0d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
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


class TestPrimitiveOp_cac252f25f8df5fd254d9711d00a191c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_cac252f25f8df5fd254d9711d00a191c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_cac252f25f8df5fd254d9711d00a191c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_573ef88642d04578fd72e214325134c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c358fed45d6cc1da90aa0e60556341e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_50f90a45fde59a080114015d64ba3c63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_f14434d7a9c4bb288f3dfac5dbd4ba5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_0675b4c542a99a44b8d3d7ae01cac5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_cb9412320436a02515088f098ab35b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_a50def626f3a693c1bdc4d4383df51c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_4fc25780e138ed8582fd11af49cc08e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_4fc25780e138ed8582fd11af49cc08e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_4fc25780e138ed8582fd11af49cc08e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_8526ac4f1aa293af593d9b65164d48fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c358fed45d6cc1da90aa0e60556341e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_c6971ea2c6df4522f08382be4e19b7e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 4], dtype='int32'),
        ]


class TestPrimitiveOp_fc302595cbcae47566bacaace7e2b68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_4f6d3b154374a9bc76d89ea0073b3a74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725, 68], dtype='int32'),
        ]


class TestPrimitiveOp_4f258148237fe032769cf370ba979364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26f73fa71aa5ea8d1c115f7e7936e739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2347, 4], dtype='int64'),
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


class TestPrimitiveOp_40621c08d1e16f78b5c0c13c978c2d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 4], dtype='int32'),
        ]


class TestPrimitiveOp_30ec09524891046ae2be7796c7789d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_e33f6f9c6470229c3a3d2ee2525e097a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069, 68], dtype='int32'),
        ]


class TestPrimitiveOp_9c9d4a7655ca773d1aadcb94fcafdf03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_222630aa894d602092f6ad9a61144149(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3109, 4], dtype='int64'),
        ]


class TestPrimitiveOp_1d29f0b94456689b651ffc6140f864ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05c68fcefee09b367cab4d626ad01c5e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_6155668a954df1721ba9058f0138328e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 4], dtype='int32'),
        ]


class TestPrimitiveOp_605eab0ef618bc0c96603f1db502928b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d815618bb9403e707958042cd23ca74c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581, 68], dtype='int32'),
        ]


class TestPrimitiveOp_2d2372b644429a2351169e5fcf3c0d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ba63ad37e8e0ea80c31b5f42e76bdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3813, 4], dtype='int64'),
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


class TestPrimitiveOp_e68e21bb2938714cb20dad4be8e4a2f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67329cbfbfe6e2e657262692edbd53a3
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


class TestPrimitiveOp_d855c01acb3dfff7555f1675cd729f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d754c6e54b0e69d20393fc36cf78f077
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_49c213619f608f2c64008d8a7aedb698(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d754c6e54b0e69d20393fc36cf78f077
    def get_inputs(self):
        return [
            paddle.to_tensor([11], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_84dc899af56444be099871c44885b537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d754c6e54b0e69d20393fc36cf78f077
    def get_inputs(self):
        return [
            paddle.to_tensor([384], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b66a2e101640de751591499d7ca97d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d754c6e54b0e69d20393fc36cf78f077
    def get_inputs(self):
        return [
            paddle.to_tensor([28], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_368cc3f8371baca8d544c5f87d8edfdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d754c6e54b0e69d20393fc36cf78f077
    def get_inputs(self):
        return [
            paddle.to_tensor([77], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_2c383cde071b770f0c99d4b54bd5c870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[152], dtype='int64'),
        ]


class TestPrimitiveOp_27b007ac5a8328cbcadbe8257e4fa6e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[100], dtype='int64'),
        ]


class TestPrimitiveOp_67f816c9481d41a988d0f59dfca06b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([100, 152, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0eadbf4bf2273f95c41056fa9533fc42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([100, 152, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc4eea0a60981b6d21b74d94be0a2423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[76], dtype='int64'),
        ]


class TestPrimitiveOp_4c54ae9f1926ad1e2f3ea8c3e5f37843(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[50], dtype='int64'),
        ]


class TestPrimitiveOp_a30fc38c3af14bf5d3b1f0b4ff187b89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([50, 76, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d38b0268e0eedc36b03deb997346fbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([50, 76, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69f7bca51c86bfefa404f5563702a9b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[38], dtype='int64'),
        ]


class TestPrimitiveOp_1db89c6758a94975e33fd7969904c972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype='int64').reshape([25]),
        ]


class TestPrimitiveOp_c031e0f3babfd39b352634ad69367ecc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([25, 38, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97e84270e87bbe3e0fabbe9ffff7f544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([25, 38, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07eee576f977a434a9e5afee1bee8f4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype='int64').reshape([19]),
        ]


class TestPrimitiveOp_1133e468407977ab877c4d48248ddd70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype='int64').reshape([13]),
        ]


class TestPrimitiveOp_6ba3eb8fd4f55adc8431c01bcf39ccf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([13, 19, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_81e6c702b5d8bde85822e868dedbbd86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([13, 19, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_191e71de83910e9f8176910b647b2a0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64').reshape([10]),
        ]


class TestPrimitiveOp_cb8d87c6b8c2bb5eadd5a99dc79a5a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6], dtype='int64').reshape([7]),
        ]


class TestPrimitiveOp_c96649b27bf408a7e470f0a40a5443b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
    def get_inputs(self):
        return [
            paddle.uniform([7, 10, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_095a63a158242410d4638a32e9f92011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_45adb4fdec2ad4a9a74e75d9fcc3b8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d1c53dc389badbafed1347ab6297f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_5bcb55c05eda23ed5d44ac6f344028ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f16aeb04493caaa635761e0dc5bd152b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_13b51cb11cea184d3367874731b08536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4888e346918d3535335f3532e1b10a11
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43365156650543213, 0.14819391071796417, 0.36823219060897827, 0.11995603144168854, 0.29551056027412415, 0.16545988619327545, 0.23600615561008453, 0.33373603224754333, 0.3894231617450714, 0.2624466121196747, 0.3304726183414459, 0.04686542972922325, 0.17661075294017792, 0.16494564712047577, 0.3988954424858093, 0.017083849757909775, 0.04391861706972122, 0.3052978515625, 0.18097592890262604, 0.31674817204475403], dtype='float32').reshape([20]),
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


class TestPrimitiveOp_0675b4c542a99a44b8d3d7ae01cac5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_f14434d7a9c4bb288f3dfac5dbd4ba5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_50f90a45fde59a080114015d64ba3c63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_d8a6ab8c2d3f75e4733c332dda561fe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
        ]


class TestPrimitiveOp_145a6d15ba2fcc9ee0ff11e2c73bc351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9e63f55bcbcd7c88aaf555400a977e66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
        ]


class TestPrimitiveOp_9abb3744d32103d3e11fc0ca944baffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7edf8852a38219f0117c4cceeb03e285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2100, 4], dtype='int64'),
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


class TestPrimitiveOp_750ccdc547e22b34b01f644d87c1fa3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_5939a0669bd5e43bad47ff5a9669df72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[72], dtype='int64'),
        ]


class TestPrimitiveOp_66debda542f81ca4387c9f7360b95451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
        ]


class TestPrimitiveOp_a43073ccfa34f758c88475da88a7881a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78fa2a50088b0f2c44f61257ed686456
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype='int64').reshape([18]),
        ]


class TestPrimitiveOp_f2af3118e4c2c39c588c3f5815919907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2af3118e4c2c39c588c3f5815919907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cceba588efb2cce6310a59179db9450
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


class TestPrimitiveOp_4fc25780e138ed8582fd11af49cc08e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_445147eb008d0af23b194c80d911dcfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_826a90626ff39f464ae13a6660316480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_8d1c53dc389badbafed1347ab6297f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614fd400b8b2ac377a78a522c72b3e14
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


class TestPrimitiveOp_93e16a3b1bdb9d3ee945acf7ad6bba03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 4], dtype='int32'),
        ]


class TestPrimitiveOp_17f7717e92a9479f0ca6239434e22ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b3da5ca116a3c8cd2602d1697852c4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_bf7d6840c31f367211ed55eec243d89f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c785908ecb825189e48322e59b25894b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400, 68], dtype='int32'),
        ]


class TestPrimitiveOp_f5f462b90e5dda6344ab230f30597613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf33482bd49195c5fe8c0182d505424
    def get_inputs(self):
        return [
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6e7a194d677645aeca94fdfd8bdac32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0038e1b2949050eeb1c879e5d3c72857
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4231, 4], dtype='int64'),
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


class TestPrimitiveOp_0675b4c542a99a44b8d3d7ae01cac5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_f14434d7a9c4bb288f3dfac5dbd4ba5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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


class TestPrimitiveOp_50f90a45fde59a080114015d64ba3c63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74168d360d4449d2f23e4a840a6867b7
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