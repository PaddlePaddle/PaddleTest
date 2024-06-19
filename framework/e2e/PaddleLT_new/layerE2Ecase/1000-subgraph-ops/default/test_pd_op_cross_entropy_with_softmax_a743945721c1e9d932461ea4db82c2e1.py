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



class PrimitiveOp_134762e5a722f81a7936be279bc39270(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe2dfd22be41437c454b0e467c5e56c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([1508, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1508, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_fe2dfd22be41437c454b0e467c5e56c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([1508, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1508, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_d874bde323753e493cc54738462952f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2377, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2377, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_d874bde323753e493cc54738462952f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2377, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2377, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_0dc54c1754933aba7499985e1525ce46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b07e3feaa086e7aa0ce47d6b3a0d2892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dc54c1754933aba7499985e1525ce46
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
        ]


class TestPrimitiveOp_e940bd255be0947498dd82cd795f2370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dc54c1754933aba7499985e1525ce46
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
        ]


class TestPrimitiveOp_0863e27d39f9e78080b1fa96a2206669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2015, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2015, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_0863e27d39f9e78080b1fa96a2206669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2015, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2015, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_5dca1a70a89648f66e2062c45f4cb611(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[20, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_896868458deda8dd9569d57d0e2107c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dca1a70a89648f66e2062c45f4cb611
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
        ]


class TestPrimitiveOp_f0457b745c33253f2a72bf674e43beb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dca1a70a89648f66e2062c45f4cb611
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
        ]


class PrimitiveOp_00f204ea65841ce67bf69ea3ce82366f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e1fb97e7b8184188f195e9996951174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00f204ea65841ce67bf69ea3ce82366f
    def get_inputs(self):
        return [
            paddle.uniform([1830, 4, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1830, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_2e1fb97e7b8184188f195e9996951174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00f204ea65841ce67bf69ea3ce82366f
    def get_inputs(self):
        return [
            paddle.uniform([1830, 4, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1830, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_d2f90d6bb04c5576f5575008cb40196f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a0526c4def39a4424141967581c5818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f90d6bb04c5576f5575008cb40196f
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
        ]


class TestPrimitiveOp_07a463d486fb55fb16173663d0edebe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f90d6bb04c5576f5575008cb40196f
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
        ]


class PrimitiveOp_a81929e3bccea64c9ff287d8afbf76fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2434, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2434, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e790ef594139ca5b798c28a6eb7e27d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a81929e3bccea64c9ff287d8afbf76fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int64'),
        ]


class TestPrimitiveOp_879583e4afefc1ca4e5321e5c6b64a95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([3039, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[3039, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_879583e4afefc1ca4e5321e5c6b64a95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([3039, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[3039, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_e9dbab7854da4261243e3534d22450b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2046, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2046, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_e9dbab7854da4261243e3534d22450b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2046, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2046, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_cd58cfdf22882e6bd916b25c2fada4cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([5498, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[5498, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_cd58cfdf22882e6bd916b25c2fada4cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([5498, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[5498, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_072c4e843ac954db84d96455d506e7d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1074, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_072c4e843ac954db84d96455d506e7d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1074, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_6d5502bb7617f00c1f4737e9688c8fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([1773, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1773, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_6d5502bb7617f00c1f4737e9688c8fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([1773, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1773, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_de4479160ab9bf13d54074a41f81238d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e1f876241a4cd4636a602f0dec8a4f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de4479160ab9bf13d54074a41f81238d
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
        ]


class TestPrimitiveOp_8001fe60843011122f93e7776c15d0ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de4479160ab9bf13d54074a41f81238d
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
        ]


class PrimitiveOp_ca1f8fefc1540fc5c853b5462cd83965(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8732, 21], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8732, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2bcaf5edd829a7279ec626fde7191329(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca1f8fefc1540fc5c853b5462cd83965
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 21], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int64'),
        ]


class TestPrimitiveOp_e92312783040ecd6cb2efc93bafabf2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([4224, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[4224, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_e92312783040ecd6cb2efc93bafabf2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([4224, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[4224, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_1f749fef49e7a365779404857903c94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([4657, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[4657, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_1f749fef49e7a365779404857903c94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([4657, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[4657, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_496222b6ec10454242b2427368909b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([3770, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[3770, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_496222b6ec10454242b2427368909b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([3770, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[3770, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_e32ef3b480e2da442b34fc185f2dc47a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_161585904a8ea0543cd8e2a6008fe25c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e32ef3b480e2da442b34fc185f2dc47a
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[36, 1], dtype='int64'),
        ]


class TestPrimitiveOp_161585904a8ea0543cd8e2a6008fe25c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e32ef3b480e2da442b34fc185f2dc47a
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[36, 1], dtype='int64'),
        ]




if __name__ == '__main__':
    unittest.main()