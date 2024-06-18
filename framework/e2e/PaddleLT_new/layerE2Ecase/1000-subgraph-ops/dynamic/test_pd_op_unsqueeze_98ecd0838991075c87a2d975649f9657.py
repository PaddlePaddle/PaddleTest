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



class PrimitiveOp_95058d5dcd915b9acabe65922ebe3560(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad86cb17cfbb39760cb8455f80c1a127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([1524], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d4b83b12c11afc7908759a4e34daf03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1524, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d4b83b12c11afc7908759a4e34daf03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1524, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c716032efc85714ea7fabddcc6a7c63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([2340], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b97fbba4b47bd55295376e037b46853a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2340, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b97fbba4b47bd55295376e037b46853a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2340, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_df5a3467abfd10aa3334c76b923a3e86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([2047], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2cef75c4ee3e855076bd8ee5fe46ff68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2047, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2cef75c4ee3e855076bd8ee5fe46ff68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2047, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_951389892ecb42c5f171db1527da2c7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([1813], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e527a8fa56ac5037d801554745db9e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1813, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e527a8fa56ac5037d801554745db9e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1813, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fbd5dee0508b100f321bd61334a94cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([3061], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8eb5cd930296e2fcd47857fa2b875603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3061, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8eb5cd930296e2fcd47857fa2b875603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3061, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0c72d232568b92289af49137ad6a4856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([2062], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f89482e59fa66950f1dad24ed993b6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2062, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1f89482e59fa66950f1dad24ed993b6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2062, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d36cc226ec5983b5866be7c8faee6d1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b3d2ba5f85594082896621bed7616374(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5ef9af61670833ec30ea8f6fd617cc59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1da93f26179ee30cb755c891fd30e869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef9af61670833ec30ea8f6fd617cc59
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1da93f26179ee30cb755c891fd30e869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef9af61670833ec30ea8f6fd617cc59
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_99317587d98f01a42939b44ec73a52c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([5526], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e17d5092756de5fe79a08bb94a10e0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5526, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e17d5092756de5fe79a08bb94a10e0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5526, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b063b88003684dafc193712342f2383f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([1071], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00d7be10ed81dc95770fa877e3e38d1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1071, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00d7be10ed81dc95770fa877e3e38d1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1071, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9f5fd7fc3b8dfef3966c70ab0b4cab39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.8947368264198303, -0.7894737124443054, -0.6842105388641357, -0.5789473652839661, -0.4736842215061188, -0.3684210479259491, -0.2631579041481018, -0.15789473056793213, -0.05263157933950424, 0.05263157933950424, 0.15789473056793213, 0.2631579041481018, 0.3684210479259491, 0.4736842215061188, 0.5789473652839661, 0.6842105388641357, 0.7894737124443054, 0.8947368264198303, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc0e6e12456831499892d13d0adab550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.931034505367279, -0.8620689511299133, -0.7931034564971924, -0.7241379022598267, -0.6551724076271057, -0.5862069129943848, -0.517241358757019, -0.4482758641242981, -0.37931033968925476, -0.3103448152542114, -0.24137930572032928, -0.17241379618644714, -0.1034482792019844, -0.03448275849223137, 0.03448275849223137, 0.1034482792019844, 0.17241379618644714, 0.24137930572032928, 0.3103448152542114, 0.37931033968925476, 0.4482758641242981, 0.517241358757019, 0.5862069129943848, 0.6551724076271057, 0.7241379022598267, 0.7931034564971924, 0.8620689511299133, 0.931034505367279, 1.0], dtype='float32').reshape([30]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d45523d5e92d65ebab82ba2f51caa88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef9af61670833ec30ea8f6fd617cc59
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d45523d5e92d65ebab82ba2f51caa88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef9af61670833ec30ea8f6fd617cc59
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e8ab55646e12cf245ee0942e74c43c4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([1760], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf20a1760b1dd11038f26a1d4c16a8d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1760, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf20a1760b1dd11038f26a1d4c16a8d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1760, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_290e25b524de3f966862af904eb4507d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.8571428656578064, -0.7142857313156128, -0.5714285969734192, -0.4285714328289032, -0.2857142984867096, -0.1428571492433548, 5.551115123125783e-17, 0.1428571492433548, 0.2857142984867096, 0.4285714328289032, 0.5714285969734192, 0.7142857313156128, 0.8571428656578064, 1.0], dtype='float32').reshape([15]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6161bcdae3ab392accd5c093f2b0fabb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_17276fff1c8d9ed990bd0afdf4b8cd40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef9af61670833ec30ea8f6fd617cc59
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_17276fff1c8d9ed990bd0afdf4b8cd40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef9af61670833ec30ea8f6fd617cc59
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_907376a5225d2c4da812659cfbad8b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([4204], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd77bcf8f88be45e19039e297ead2e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4204, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd77bcf8f88be45e19039e297ead2e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4204, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04b39b86c0d8ee95f6a0f65880e27d8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9130434989929199, -0.8260869383811951, -0.739130437374115, -0.6521739363670349, -0.5652173757553101, -0.47826087474823, -0.3913043439388275, -0.30434781312942505, -0.21739129722118378, -0.1304347813129425, -0.043478261679410934, 0.043478261679410934, 0.1304347813129425, 0.21739129722118378, 0.30434781312942505, 0.3913043439388275, 0.47826087474823, 0.5652173757553101, 0.6521739363670349, 0.739130437374115, 0.8260869383811951, 0.9130434989929199, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4870ec48481694e6a35957e7daa09ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e279a69105495e7ad7527d023f44e226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef9af61670833ec30ea8f6fd617cc59
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e279a69105495e7ad7527d023f44e226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ef9af61670833ec30ea8f6fd617cc59
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bb9268f11f40abee77ecd6bbe5483dbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([4680], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aee10b0e58bae36811512a313d719ebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4680, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aee10b0e58bae36811512a313d719ebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4680, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6118f0a41afe0a61b202005397d63ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95058d5dcd915b9acabe65922ebe3560
    def get_inputs(self):
        return [
            paddle.uniform([3778], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c7b9ee42acb303b86c2a6059471da8a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3778, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c7b9ee42acb303b86c2a6059471da8a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fe5b80c663b2286b25647bc6a1db838
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3778, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()