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



class PrimitiveOp_67f15bd6baa68c2a77c796d98377096f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [11, 1, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afc19be982193cebbdb4a91406edda45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f15bd6baa68c2a77c796d98377096f
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_96c243e97ce0e7c03889f98c367e1b03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [43, 1, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2414be184cbce556a3461f9706a4b66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96c243e97ce0e7c03889f98c367e1b03
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c2414be184cbce556a3461f9706a4b66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96c243e97ce0e7c03889f98c367e1b03
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_afc19be982193cebbdb4a91406edda45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f15bd6baa68c2a77c796d98377096f
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_afc19be982193cebbdb4a91406edda45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f15bd6baa68c2a77c796d98377096f
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c2414be184cbce556a3461f9706a4b66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96c243e97ce0e7c03889f98c367e1b03
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4f447b1fdf12e371a71840a4b8d36950(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [1, 64, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a98d92c41ecde54b07d43e6a5840946c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f447b1fdf12e371a71840a4b8d36950
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3088676eaea55f3b712cee9ba7799908(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [1, 512, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc1fd6710ff69cd3b065354dd282a0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3088676eaea55f3b712cee9ba7799908
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c2414be184cbce556a3461f9706a4b66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96c243e97ce0e7c03889f98c367e1b03
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_75c5249c6c3cecbde84ed615942ac538(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [1, 192, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77d5078706a49715dbcfcc5086a9f8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c5249c6c3cecbde84ed615942ac538
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a98d92c41ecde54b07d43e6a5840946c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f447b1fdf12e371a71840a4b8d36950
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_76c35c05068f12663449e4012fa17d44(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [1, 256, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2bc2c5309b7b2f429bb6655fa658fe59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76c35c05068f12663449e4012fa17d44
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2bc2c5309b7b2f429bb6655fa658fe59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76c35c05068f12663449e4012fa17d44
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bc26c9e5b0c8ada5746623b61e92b6ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [1, 128, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b116d4fc9dbf0e879d10c53ff93add5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc26c9e5b0c8ada5746623b61e92b6ca
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_afc19be982193cebbdb4a91406edda45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f15bd6baa68c2a77c796d98377096f
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc1fd6710ff69cd3b065354dd282a0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3088676eaea55f3b712cee9ba7799908
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc1fd6710ff69cd3b065354dd282a0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3088676eaea55f3b712cee9ba7799908
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc1fd6710ff69cd3b065354dd282a0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3088676eaea55f3b712cee9ba7799908
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_79cd7e85875900df81c1623c125c941e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        input_0 = [1, 2048, 1, 1]
        return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fbb43794f172f3a6f8442db8dbdc1ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd7e85875900df81c1623c125c941e
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc1fd6710ff69cd3b065354dd282a0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3088676eaea55f3b712cee9ba7799908
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc1fd6710ff69cd3b065354dd282a0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3088676eaea55f3b712cee9ba7799908
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc1fd6710ff69cd3b065354dd282a0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3088676eaea55f3b712cee9ba7799908
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9fbb43794f172f3a6f8442db8dbdc1ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd7e85875900df81c1623c125c941e
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_afc19be982193cebbdb4a91406edda45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67f15bd6baa68c2a77c796d98377096f
    def get_inputs(self):
        return [
            paddle.to_tensor([11, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5b116d4fc9dbf0e879d10c53ff93add5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc26c9e5b0c8ada5746623b61e92b6ca
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2bc2c5309b7b2f429bb6655fa658fe59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76c35c05068f12663449e4012fa17d44
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c2414be184cbce556a3461f9706a4b66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96c243e97ce0e7c03889f98c367e1b03
    def get_inputs(self):
        return [
            paddle.to_tensor([43, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc1fd6710ff69cd3b065354dd282a0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3088676eaea55f3b712cee9ba7799908
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc1fd6710ff69cd3b065354dd282a0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3088676eaea55f3b712cee9ba7799908
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5b116d4fc9dbf0e879d10c53ff93add5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc26c9e5b0c8ada5746623b61e92b6ca
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()