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


class TestPrimitiveOp_64c62201d6c2495a28ea1ec0ab7fa069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
        ]


class TestPrimitiveOp_274b8b30907a710f9e3671ccfa648d6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
        ]


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


class TestPrimitiveOp_cedd7814ea4ba7496415d48cd42244b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1762, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1762, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_cedd7814ea4ba7496415d48cd42244b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1762, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1762, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_c281d0bd37be59eaa83b122a526bf195(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([5522, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[5522, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_c281d0bd37be59eaa83b122a526bf195(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([5522, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[5522, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_612065ee81c9dcac2e2a55eb9f72fe07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
        ]


class TestPrimitiveOp_612065ee81c9dcac2e2a55eb9f72fe07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
        ]


class TestPrimitiveOp_535867ca571a5d8e48a84968d0e0ab5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1760, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_535867ca571a5d8e48a84968d0e0ab5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1760, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_a1a4c0cd7568e115aeafe3c07332bb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
        ]


class TestPrimitiveOp_376cdccac462ef56dd631852432b95fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
        ]


class TestPrimitiveOp_216a6ceeae063b980def0ecf0e782951(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1522, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1522, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_216a6ceeae063b980def0ecf0e782951(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1522, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1522, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_3c8549c74b7847d10e46dfa23e51f0e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
        ]


class TestPrimitiveOp_700e9668046a86c03924f2dc8cbb7a80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
        ]


class TestPrimitiveOp_33cfaf9711b3495e03286f22d16bf153(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([2074, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2074, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_33cfaf9711b3495e03286f22d16bf153(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([2074, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2074, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_c6f9752db027d23eb140f91b9da8bfbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([4734, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4734, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_c6f9752db027d23eb140f91b9da8bfbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([4734, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4734, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_b8bab24e99fffcaaccc87578e6f1b37a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
        ]


class TestPrimitiveOp_3f3c91b62df5045388fcb9d9beeb7865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1074, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_3f3c91b62df5045388fcb9d9beeb7865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1074, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_dd9ddae2adc5e7a8804d5df234b8e942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([2332, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2332, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_dd9ddae2adc5e7a8804d5df234b8e942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([2332, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2332, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_12158c128e263605679ee1be26c658c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([3051, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3051, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_12158c128e263605679ee1be26c658c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([3051, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3051, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_c194e94d19b662d8b56a003043176cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([3870, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3870, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_c194e94d19b662d8b56a003043176cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([3870, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3870, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_03e51daab750fd342f921330fd8b0f3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
        ]


class TestPrimitiveOp_228aa5e04bc1551cbf5eb9a640adbea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5273e75dbc02ba9504ef338011be72b7
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
        ]


class TestPrimitiveOp_812387a070818ecca3699b427c8315ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 21], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
        ]


class TestPrimitiveOp_76f45a68a54b46811a75c7e566e4ab76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([2111, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2111, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_76f45a68a54b46811a75c7e566e4ab76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([2111, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2111, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_7da69c3db0c085ac71a6df03d787f085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([4141, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4141, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_7da69c3db0c085ac71a6df03d787f085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbbbd5be6171686bf8896f7bedfb8897
    def get_inputs(self):
        return [
            paddle.uniform([4141, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4141, 4, 1], dtype='int64'),
        ]




if __name__ == '__main__':
    unittest.main()