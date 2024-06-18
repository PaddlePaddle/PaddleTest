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



class PrimitiveOp_ee00e9f5a439d12faf3a32b5eb6aaf41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.pow(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_418805646049ecb733229191a0e59377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee00e9f5a439d12faf3a32b5eb6aaf41
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7c73ab697d8d1076629c251eb9e30bb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee00e9f5a439d12faf3a32b5eb6aaf41
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.066104769706726], [0.9551755785942078], [0.8827231526374817], [0.5879164338111877], [0.48536917567253113], [0.4193213880062103]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor(3.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_418805646049ecb733229191a0e59377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee00e9f5a439d12faf3a32b5eb6aaf41
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a3e8644c68c09091251e02ea29fc1670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee00e9f5a439d12faf3a32b5eb6aaf41
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.12546059489250183], [1.2690961360931396], [0.7117348909378052], [0.63327556848526], [0.278152197599411], [0.7673402428627014]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor(3.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.pow(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80203cf811e26c3a8742eb7cf19de11d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_80203cf811e26c3a8742eb7cf19de11d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6558af035597b15c71b26259cccf9a19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.pow(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77e269c5e97ec44a8975e60aee2b3052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_77e269c5e97ec44a8975e60aee2b3052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a62e30d225f40663f58966b32c8c6038(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a62e30d225f40663f58966b32c8c6038(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bcfa91226a5c056cb7f2c0d81d57ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bcfa91226a5c056cb7f2c0d81d57ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3cdcce6868523c37a6148d0cf28cf887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3cdcce6868523c37a6148d0cf28cf887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bb08d4a51549b835a07fdc63fa5150d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bb08d4a51549b835a07fdc63fa5150d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_504340067f8d0c234db0cddf652cd9f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_527e1c9de059951e9654caa6a9dd7377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee00e9f5a439d12faf3a32b5eb6aaf41
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_504340067f8d0c234db0cddf652cd9f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7622f19c840c4bd9d8e80de5cc1d0687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0ab789c38f563be20bfc29cd0a89f873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0a548605c967443a4d614b73591b73b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_be79861106c58c4ae52b357db27a0f67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ebee80ce2fe3c67f89300e6516c14cc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_651fc3a2eef426ef34700909dccd27d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_aa169eaae677e0d6bb49da02a8231b21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_aa169eaae677e0d6bb49da02a8231b21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1da054c1a3001caef6811191cd261770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1da054c1a3001caef6811191cd261770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ce2fd742951db3727daaa9f3cfbadee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ce2fd742951db3727daaa9f3cfbadee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4fdf38fac93354b006a259ffd72869d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4fdf38fac93354b006a259ffd72869d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3561196acc55fd027de2200732457ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_77e269c5e97ec44a8975e60aee2b3052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_77e269c5e97ec44a8975e60aee2b3052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a62e30d225f40663f58966b32c8c6038(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a62e30d225f40663f58966b32c8c6038(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bcfa91226a5c056cb7f2c0d81d57ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0bcfa91226a5c056cb7f2c0d81d57ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3cdcce6868523c37a6148d0cf28cf887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3cdcce6868523c37a6148d0cf28cf887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_aa169eaae677e0d6bb49da02a8231b21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_aa169eaae677e0d6bb49da02a8231b21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1da054c1a3001caef6811191cd261770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1da054c1a3001caef6811191cd261770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ce2fd742951db3727daaa9f3cfbadee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ce2fd742951db3727daaa9f3cfbadee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4fdf38fac93354b006a259ffd72869d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c4fdf38fac93354b006a259ffd72869d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebc994ec435ed33bfb6614f5cc70122
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_90d0e7f4c1e05db6c1a556d2a6942423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_999082198cdfaa6e505a4c42e5e02c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e0cebc4fed92c332b8030149a714bfc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fd2aa0119ab280916ca8115e0fd665ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f8d684f91364e2dc9e9a3ccdda787072(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92007c52119b3998c338fbdc3f93a9d2
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2.0, dtype='float32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()