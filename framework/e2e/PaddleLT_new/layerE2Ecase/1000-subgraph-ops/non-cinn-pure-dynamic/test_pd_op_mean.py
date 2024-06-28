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



class PrimitiveOp_1585dda4138c3b692c2230ab179c53ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2, 3], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ee259919af2bffbf8f3f3f1e035b492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9b2fd8951b0ec879d023d2d386f664f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b046274fecc1f0201d4d21655e8d022d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4271e7b9e9e9c82d1c27c172218c096e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5184767535196a953a7cbf9f7d252794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8165f7c0f708aaca8105f70caaa65a40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8cedb5d4948836c201d091804381f5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1757ca259e3758cd391c721301d9c4fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8923592dd777e7525b44aac22332de4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3a94662d587b3d3d6bb16612e851f250(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e206f3e2a78182f8ab2f5e29ded78db3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b5513015b9ca67fa6a288d6843c725f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [-1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22e562d26d2adcf0429ce83de86c95d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5513015b9ca67fa6a288d6843c725f2
    def get_inputs(self):
        return [
            paddle.uniform([1762, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ae4fa6dfa11566e0d0ecf1b7ca1671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb4004159d4c924b75ea4ab7a8ae63c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ccd966085de0ff53a74c08c7eb4cfc9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b8c7c4a7b95912831be6114f6d98b2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8943f9a3dd99070a08e41d8dcdc9813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5513015b9ca67fa6a288d6843c725f2
    def get_inputs(self):
        return [
            paddle.uniform([5522, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4bfa9392203c4f023072df1aaa09cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1eab852e141297be90a96cbcc422366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4a4a6430f0dd7f125d9889b801991d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_229353080870aede826e194cfa85f6b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be7ff6c90721844299f8ec4585d017a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc9684b46985834d3256ce36b42fb0df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [1], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1b9734c537a548be579c961f6469f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1b9734c537a548be579c961f6469f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c492f4e5e84fab991edd8c1d5d15d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c492f4e5e84fab991edd8c1d5d15d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a121e7925079225a82383d681a25cc8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a121e7925079225a82383d681a25cc8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5506b529c91dcccdedfe256590cb14c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5506b529c91dcccdedfe256590cb14c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c198bd7f30f6036204a837f5caeacffa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [], False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_526486e2b74fbae6e4d137fb2d7a33c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c198bd7f30f6036204a837f5caeacffa
    def get_inputs(self):
        return [
            paddle.to_tensor([1.3266841173171997, 1.7957855463027954, 2.3504652976989746, 1.5122127532958984, 1.211737871170044, 1.2600581645965576], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_8140195382d150e43b464151df8b97dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8140195382d150e43b464151df8b97dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e93f3acc0ef11c6dafde622532fcbe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e93f3acc0ef11c6dafde622532fcbe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e66f34d0a683ec63f6affb97fb062c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e66f34d0a683ec63f6affb97fb062c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b085e44333b5622bb81ee6ad0f53aded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b085e44333b5622bb81ee6ad0f53aded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6490178500279fd53742366c4ca9380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_def8d141b5b3e7b1a8ef54c02249a9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5513015b9ca67fa6a288d6843c725f2
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f3f956b02d802d1506e5b7469b9c8ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30b818f0d9185dcb576801ce27947f70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5c04941447d85fd760468735a88640b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8140195382d150e43b464151df8b97dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8140195382d150e43b464151df8b97dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e93f3acc0ef11c6dafde622532fcbe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e93f3acc0ef11c6dafde622532fcbe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e66f34d0a683ec63f6affb97fb062c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e66f34d0a683ec63f6affb97fb062c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b085e44333b5622bb81ee6ad0f53aded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b085e44333b5622bb81ee6ad0f53aded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e578f39b74da27851068043bdb1dd977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5513015b9ca67fa6a288d6843c725f2
    def get_inputs(self):
        return [
            paddle.uniform([1522, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f4d086b43dbd1a1c7a8f0998dd926ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9bfb84b89fb6d3269083c140ee69cc1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46a5731546e28e4993aeced2bcf18f7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e86b950d3f840ff7400ca7ce0f51c107(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.mean(input_0, [2], True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_020ec48637bec71ea35a0aa9dd3a9798(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e86b950d3f840ff7400ca7ce0f51c107
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f9552afb83aa1254a5b6caa200aa93c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c5b3c8af892c03c26742625874025b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_286eae361f9f2b2acc73f23b2d6a079e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a56f998ef3704d1d83590f9afd34cd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c66b1486a7095ae5939adbeb76de841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6a18a93a14767a8395dfd0a2393afb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f17f3b5c0a6762ccf2a13ef1ab677b2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5513015b9ca67fa6a288d6843c725f2
    def get_inputs(self):
        return [
            paddle.uniform([2074, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dab9fbbf8d5f6ec049327d797aa71818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ad6c392cdbb673dd09ab63c4869781b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e908a1d39300e8c3883540316880e678(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5513015b9ca67fa6a288d6843c725f2
    def get_inputs(self):
        return [
            paddle.uniform([4734, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ef4d13cc725dafb68bd0b9179064f29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2caca2d20f059812c0f07b6e1e418381(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5513015b9ca67fa6a288d6843c725f2
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_afddf9b507118499b3d4b37bc28eb50b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2641d964d0708faaa55d05c6b7fd4fdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99ef4604f5856f8eb93b6de73149c595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ca5ccf2484af2bb2a5bc23c7140ea14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca9991e627041e0af3b7f9370428fc27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7a0521edcaf61be98366a09490f3942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1b9734c537a548be579c961f6469f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1b9734c537a548be579c961f6469f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c492f4e5e84fab991edd8c1d5d15d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c492f4e5e84fab991edd8c1d5d15d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a121e7925079225a82383d681a25cc8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a121e7925079225a82383d681a25cc8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5506b529c91dcccdedfe256590cb14c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5506b529c91dcccdedfe256590cb14c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa2a0ec0fbfdd8e09af706e982ec674f
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb8898bb2ac152d7da845dd91c5dbd68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c2c24dada79e7747a8d1e02321e836e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5513015b9ca67fa6a288d6843c725f2
    def get_inputs(self):
        return [
            paddle.uniform([2332, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe9a310c85173a0a1a7cfa6a6b76bd19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5f385e4325230bebc3102ad34282d56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5513015b9ca67fa6a288d6843c725f2
    def get_inputs(self):
        return [
            paddle.uniform([3051, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88b31a2a12b6bc5edc1764f17074d99a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5513015b9ca67fa6a288d6843c725f2
    def get_inputs(self):
        return [
            paddle.uniform([3870, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2173f3c563671903ae193eca3edc2f6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef58696e8743e92da8cb5b77f6df7793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ac24573ba280a6b895948a0b43af0a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3a2ee6fe83f9ab7556d2333f86243974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8b5e5cba98c477985405a5ad172f409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c3f8f0d76d5a20d88cecbd67fbcfb7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e86b950d3f840ff7400ca7ce0f51c107
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06bd30cbceff99e6c78ef8ef9dd0c22a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d72895e5b2b34f7134d33d0422b9944e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7823b11f9235631fc146a238f8eec72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfe2d65374fbef6c68eba5545cab1c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94b3bcb68578911434a267644c5c8545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_408fcbad90ec74231ef27b9337ac8b78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f243f4bdec71a127196f6d3793b12ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edcba021bf293c6fc458a2cb5466da72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5513015b9ca67fa6a288d6843c725f2
    def get_inputs(self):
        return [
            paddle.uniform([2111, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0738e233bce5229dbef55b508e731699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2da140b0323ec92ac9ff6c433a3eb5a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40c1e3bee9242cd80752d4537311b159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b70edb8414bd7c87fe581e34e97347dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f987fc08194ef946100ea3d4c4ba4873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_109a6f42ed12b3d10b02b88e7ac9b62b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e86b950d3f840ff7400ca7ce0f51c107
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dd99dd1c63fd5c6c4e8bf7643190b0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7dfced26a44d6199c15c5eaaf60fc730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e86b950d3f840ff7400ca7ce0f51c107
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79ed765671c08c3e8bcba7120c47c479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5513015b9ca67fa6a288d6843c725f2
    def get_inputs(self):
        return [
            paddle.uniform([4141, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3da31b2dab52db0d4e1622c6c475aafd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e86b950d3f840ff7400ca7ce0f51c107
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d3bdd346e17d16d2aecfa309e1607a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1585dda4138c3b692c2230ab179c53ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()