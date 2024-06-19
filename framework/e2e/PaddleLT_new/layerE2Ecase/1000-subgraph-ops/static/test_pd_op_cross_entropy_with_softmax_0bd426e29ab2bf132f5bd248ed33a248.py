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



class PrimitiveOp_672c02f34e281a7aae0aa3eecb56c29b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1542, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b5e6019a0ceedd5ce7b73db6c871afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672c02f34e281a7aae0aa3eecb56c29b
    def get_inputs(self):
        return [
            paddle.uniform([1542, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1542, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_9b5e6019a0ceedd5ce7b73db6c871afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672c02f34e281a7aae0aa3eecb56c29b
    def get_inputs(self):
        return [
            paddle.uniform([1542, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1542, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_0d312a42b25e53b64c1abd7765cc42d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[2361, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a764c558c9ec3826d5493f800a219918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d312a42b25e53b64c1abd7765cc42d4
    def get_inputs(self):
        return [
            paddle.uniform([2361, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2361, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_a764c558c9ec3826d5493f800a219918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d312a42b25e53b64c1abd7765cc42d4
    def get_inputs(self):
        return [
            paddle.uniform([2361, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2361, 4, 1], dtype='int64'),
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


class PrimitiveOp_202de086a81170a2c78bca7ce20ae1de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[2053, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa9af50a4f6e3e31ffcfbe8c7cab6c18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202de086a81170a2c78bca7ce20ae1de
    def get_inputs(self):
        return [
            paddle.uniform([2053, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2053, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_aa9af50a4f6e3e31ffcfbe8c7cab6c18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202de086a81170a2c78bca7ce20ae1de
    def get_inputs(self):
        return [
            paddle.uniform([2053, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2053, 4, 1], dtype='int64'),
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


class PrimitiveOp_4d965e725c614f3e974dd34cda3a86bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 4, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1825, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53859c06880ecf98db9bc0be5e3e61fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d965e725c614f3e974dd34cda3a86bb
    def get_inputs(self):
        return [
            paddle.uniform([1825, 4, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1825, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_53859c06880ecf98db9bc0be5e3e61fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d965e725c614f3e974dd34cda3a86bb
    def get_inputs(self):
        return [
            paddle.uniform([1825, 4, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1825, 4, 1], dtype='int64'),
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


class PrimitiveOp_6f1662ceb98fd7bc586b16f8f7e6b781(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[3087, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8acd3c999284cac550da3149ebd9ab5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1662ceb98fd7bc586b16f8f7e6b781
    def get_inputs(self):
        return [
            paddle.uniform([3087, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[3087, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_8acd3c999284cac550da3149ebd9ab5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1662ceb98fd7bc586b16f8f7e6b781
    def get_inputs(self):
        return [
            paddle.uniform([3087, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[3087, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_4a91193569a18370a410a437637e1b2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[2119, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14dd47709db4379506b4d83b6a647c88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a91193569a18370a410a437637e1b2a
    def get_inputs(self):
        return [
            paddle.uniform([2119, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2119, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_14dd47709db4379506b4d83b6a647c88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a91193569a18370a410a437637e1b2a
    def get_inputs(self):
        return [
            paddle.uniform([2119, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[2119, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_5220eab2daabb1e448748c61e328f856(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[5606, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3f1134af3451f511ace8bc614839a92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5220eab2daabb1e448748c61e328f856
    def get_inputs(self):
        return [
            paddle.uniform([5606, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[5606, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_c3f1134af3451f511ace8bc614839a92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5220eab2daabb1e448748c61e328f856
    def get_inputs(self):
        return [
            paddle.uniform([5606, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[5606, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_f42b04a3d1e88b2cfe72e7eb072a05e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1036, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_381ac4d73098704d6b43b1d5d410439f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f42b04a3d1e88b2cfe72e7eb072a05e8
    def get_inputs(self):
        return [
            paddle.uniform([1036, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1036, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_381ac4d73098704d6b43b1d5d410439f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f42b04a3d1e88b2cfe72e7eb072a05e8
    def get_inputs(self):
        return [
            paddle.uniform([1036, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1036, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_1eb00cd05767e8d7dfa12177e80baa78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1809, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc42dff8ed9e0dd5bd83f0abfb8ed347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1eb00cd05767e8d7dfa12177e80baa78
    def get_inputs(self):
        return [
            paddle.uniform([1809, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1809, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_fc42dff8ed9e0dd5bd83f0abfb8ed347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1eb00cd05767e8d7dfa12177e80baa78
    def get_inputs(self):
        return [
            paddle.uniform([1809, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[1809, 4, 1], dtype='int64'),
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


class PrimitiveOp_489b1a91759881368f52fd45fb7dfb27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[4179, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb34fe8f5f59fdb8badf1970d04e0a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_489b1a91759881368f52fd45fb7dfb27
    def get_inputs(self):
        return [
            paddle.uniform([4179, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[4179, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_fb34fe8f5f59fdb8badf1970d04e0a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_489b1a91759881368f52fd45fb7dfb27
    def get_inputs(self):
        return [
            paddle.uniform([4179, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[4179, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_0814235ea389ea0f0217729a1af91577(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[4662, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad3478e825d7d911e2be84ff3bc2df9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0814235ea389ea0f0217729a1af91577
    def get_inputs(self):
        return [
            paddle.uniform([4662, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[4662, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_ad3478e825d7d911e2be84ff3bc2df9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0814235ea389ea0f0217729a1af91577
    def get_inputs(self):
        return [
            paddle.uniform([4662, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[4662, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_9c177a02edcdd5f15189f5a8a97c51e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[3857, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7101dc0589903b7a22be2f1d554af54d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c177a02edcdd5f15189f5a8a97c51e6
    def get_inputs(self):
        return [
            paddle.uniform([3857, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[3857, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_7101dc0589903b7a22be2f1d554af54d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c177a02edcdd5f15189f5a8a97c51e6
    def get_inputs(self):
        return [
            paddle.uniform([3857, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.randint(low=0, high=2, shape=[3857, 4, 1], dtype='int64'),
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