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



class PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1de404babe13028e47996d799ba5194d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1524, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1524, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_1de404babe13028e47996d799ba5194d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1524, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1524, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_4dc55bff0fb07aa518b3bd4f26fafa47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([2340, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2340, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_4dc55bff0fb07aa518b3bd4f26fafa47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([2340, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2340, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_5273e75dbc02ba9504ef338011be72b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fb9a423a1e717792e63c3000f9dea4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
        ]


class TestPrimitiveOp_5121b25a34519ce6a9d231e4300b40e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
        ]


class TestPrimitiveOp_4fe2116b0f8c7fbd60404beea31be168(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([2047, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2047, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_4fe2116b0f8c7fbd60404beea31be168(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([2047, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2047, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_1f5ae1f2dabffa9c0d4de76f4f004c36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
        ]


class TestPrimitiveOp_ab703acc05ec97d8729c7f6d0da75083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
        ]


class TestPrimitiveOp_b11b68c3a74d1975aaa1e2a4beefeecc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1813, 4, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1813, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_b11b68c3a74d1975aaa1e2a4beefeecc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1813, 4, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1813, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_8f14e7f38e9bca7e50617b3a07a5557c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
        ]


class TestPrimitiveOp_43095a5177fa0ed0aa4d1f795c646949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
        ]


class TestPrimitiveOp_0436424fb19277bbc2e6afc78965803b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int64'),
        ]


class TestPrimitiveOp_3314c981488d4a838e22236d6e2243cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([3061, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[3061, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_3314c981488d4a838e22236d6e2243cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([3061, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[3061, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_d1cb4e3c0d1acb14f573b69a9ae156ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([2062, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2062, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_d1cb4e3c0d1acb14f573b69a9ae156ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([2062, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2062, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_7971d4515e8b77795862c8db81e68d2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([5526, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[5526, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_7971d4515e8b77795862c8db81e68d2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([5526, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[5526, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_c2f0d42b59dca5521224a93b3a3d7486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1071, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1071, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_c2f0d42b59dca5521224a93b3a3d7486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1071, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1071, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_3c01219373d79ee6f860377a6821f4a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1760, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_3c01219373d79ee6f860377a6821f4a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1760, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_05c22c574c4fcb6499ee092854c0e6e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
        ]


class TestPrimitiveOp_6dab6f1a2e8964deeb60d3dc22915fcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
        ]


class TestPrimitiveOp_36006cc1288184042d456a1d1da54f6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 21], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int64'),
        ]


class TestPrimitiveOp_5ffde13210fb1997eb1865805965dd29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([4204, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[4204, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_5ffde13210fb1997eb1865805965dd29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([4204, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[4204, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_9687397ec55551e8c502ec880ed86c2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([4680, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[4680, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_9687397ec55551e8c502ec880ed86c2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([4680, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[4680, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_b4bb18a7fe0c593a355377b12cda4dc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[3778, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_b4bb18a7fe0c593a355377b12cda4dc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[3778, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_a11841ad71c5ea20c8b9fad83732700c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[36, 1], dtype='int64'),
        ]


class TestPrimitiveOp_a11841ad71c5ea20c8b9fad83732700c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[36, 1], dtype='int64'),
        ]




if __name__ == '__main__':
    unittest.main()