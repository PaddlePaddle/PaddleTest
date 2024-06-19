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



class PrimitiveOp_634229614b1c01a598326e99e4962090(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_522e36990ad9ff90c9d016e606c2ec46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([1508], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54d2ecd32ffeea840ff508b24b22fc8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1508, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54d2ecd32ffeea840ff508b24b22fc8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1508, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_833dce7134bd859ffcf3431a6ece45a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([2377], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ebeb8a1a687c900e1cb3d2be1edbcb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2377, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ebeb8a1a687c900e1cb3d2be1edbcb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2377, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6116dcb41ca2c9a86eabb5a52c760ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([2015], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8b558f3ef0e04942b2bb29bf52f51e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2015, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8b558f3ef0e04942b2bb29bf52f51e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2015, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_123e3667a62cc50f48fbe63e3a412e73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([1830], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c9b38f726b8f398d3fab134a01a0e45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1830, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c9b38f726b8f398d3fab134a01a0e45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1830, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a6f72ef477ba9b63877bf23a60cc3051(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([3039], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7423476de0c1cfc5801f10b1558a84a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3039, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7423476de0c1cfc5801f10b1558a84a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3039, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_addcb07e2dbd3a7f1dc4c18b51cb61d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([2046], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e19649a96d3eec0c45e1db46e3fb069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2046, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e19649a96d3eec0c45e1db46e3fb069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2046, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d813e0e2450ab79494b38f88a37e78c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c4a1a2ccfddf9aadb7b659b0253fcd4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_05b51286982b7aed8c7c4a9178c82c45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4058058f26a99b722d75b299b283362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b51286982b7aed8c7c4a9178c82c45
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e4058058f26a99b722d75b299b283362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b51286982b7aed8c7c4a9178c82c45
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6d9f1eb590c411b61302710c75b00a41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([5498], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ffc36b05ca1ff3ffd4be837b1c4ffc85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5498, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ffc36b05ca1ff3ffd4be837b1c4ffc85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5498, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4dd2efef6be1fb344c57cb5573413c43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([1074], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0b1a4d612c448ef27b8d830547e07897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1074, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0b1a4d612c448ef27b8d830547e07897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1074, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dec71929d443c6d969fa52b45e1fa327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.8947368264198303, -0.7894737124443054, -0.6842105388641357, -0.5789473652839661, -0.4736842215061188, -0.3684210479259491, -0.2631579041481018, -0.15789473056793213, -0.05263157933950424, 0.05263157933950424, 0.15789473056793213, 0.2631579041481018, 0.3684210479259491, 0.4736842215061188, 0.5789473652839661, 0.6842105388641357, 0.7894737124443054, 0.8947368264198303, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d6a7bf2b8cfac4d58734fb71feccbf4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.931034505367279, -0.8620689511299133, -0.7931034564971924, -0.7241379022598267, -0.6551724076271057, -0.5862069129943848, -0.517241358757019, -0.4482758641242981, -0.37931033968925476, -0.3103448152542114, -0.24137930572032928, -0.17241379618644714, -0.1034482792019844, -0.03448275849223137, 0.03448275849223137, 0.1034482792019844, 0.17241379618644714, 0.24137930572032928, 0.3103448152542114, 0.37931033968925476, 0.4482758641242981, 0.517241358757019, 0.5862069129943848, 0.6551724076271057, 0.7241379022598267, 0.7931034564971924, 0.8620689511299133, 0.931034505367279, 1.0], dtype='float32').reshape([30]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4488774757385370e2a396e8564d2b4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b51286982b7aed8c7c4a9178c82c45
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4488774757385370e2a396e8564d2b4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b51286982b7aed8c7c4a9178c82c45
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b48ffc2b7eb71acccc553adc46fdd85a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([1773], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_21d22dc26e91e00f9ccb011ea48812a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1773, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_21d22dc26e91e00f9ccb011ea48812a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1773, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e07e8faac0a425ded9dfc99d298810c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.8571428656578064, -0.7142857313156128, -0.5714285969734192, -0.4285714328289032, -0.2857142984867096, -0.1428571492433548, 5.551115123125783e-17, 0.1428571492433548, 0.2857142984867096, 0.4285714328289032, 0.5714285969734192, 0.7142857313156128, 0.8571428656578064, 1.0], dtype='float32').reshape([15]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6026c45eea8872fee0cc605e338baf26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e7bf1170100a04a568f852d1ef9788f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b51286982b7aed8c7c4a9178c82c45
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8e7bf1170100a04a568f852d1ef9788f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b51286982b7aed8c7c4a9178c82c45
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5f02ef12a349d2838d25847ea04a1d35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([4224], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2371c8a2ea57ecef5cf1c58f3a7d76c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4224, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2371c8a2ea57ecef5cf1c58f3a7d76c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4224, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_51d9d31f84c0d93d925611001a6d6b25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9130434989929199, -0.8260869383811951, -0.739130437374115, -0.6521739363670349, -0.5652173757553101, -0.47826087474823, -0.3913043439388275, -0.30434781312942505, -0.21739129722118378, -0.1304347813129425, -0.043478261679410934, 0.043478261679410934, 0.1304347813129425, 0.21739129722118378, 0.30434781312942505, 0.3913043439388275, 0.47826087474823, 0.5652173757553101, 0.6521739363670349, 0.739130437374115, 0.8260869383811951, 0.9130434989929199, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0721661764bcaa42f2d0a4b49c12d763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c2c2e70526580b495890b2e09e7baa9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b51286982b7aed8c7c4a9178c82c45
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c2c2e70526580b495890b2e09e7baa9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b51286982b7aed8c7c4a9178c82c45
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d543b8e9ca4925367fd78484c85bbc38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([4657], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_99fadf5b9169b6dc69d30eed48294717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4657, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_99fadf5b9169b6dc69d30eed48294717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4657, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_08a6d700dcd23c9c3643fa83a7fcc51e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_634229614b1c01a598326e99e4962090
    def get_inputs(self):
        return [
            paddle.uniform([3770], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9902b7388d5c6bbab365a353fc08b0b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3770, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9902b7388d5c6bbab365a353fc08b0b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8111392130f6f91e7ac858c2fc8b92ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3770, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()