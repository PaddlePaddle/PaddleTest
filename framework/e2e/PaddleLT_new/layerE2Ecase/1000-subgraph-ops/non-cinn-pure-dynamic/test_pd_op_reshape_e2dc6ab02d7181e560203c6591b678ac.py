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



class PrimitiveOp_3256feed2bb0cea5fd1736c98672718d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class PrimitiveOp_3d6fbd165e154c3918830a52b91cca77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a182d9035b86bcee8814015e3b21b6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 91], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 21504, 1, 91], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a4fe737c29f2de0610de8192d9e1d805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_71f150460cf4aeca6f1797e61a91a15f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 784, 6, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ab89296a48027a3c8439ae8d28d21342(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 192, 28, 28], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dcb22da66a9b4ec5a98fdcbe05f769eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 192, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a818699fc07f4cb5234f00baa24247dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 49, 2, 6, 32], dtype='int64').reshape([5]),
        ]


class PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7c89d9f779f7365afca873a0ae955b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([129024, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b018aba824195d89684d31f9e42c4bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([32256, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ef697ffb1c2686c5d3c071dc3f4d92d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([8064, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5a2b1d5c8035e80f87e8f6497e06dcf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([2016, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_46272270f5e01a5115d7257d0390c547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([528, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_731d9527517f9c48ae5c01b792768e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 256, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_6e144d11e9bd10c9df1560b9142f9a99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 128, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_38c54eb9e5934217d8627dea52054120(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 42, 64, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f210f68eceb8a659fa7c2cfb8d20d482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 32, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e576f72a3569c3b546f79aa95dd88996(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 11, 16, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_74cffc1d31718b358bed704a5e0370a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 256, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_fda4465368f8e8d8a6d62316cbbe149d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 128, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_cf4b028f4168cfba0c8c3f20ccc33cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 42, 64, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8d2160b98d1913b003d9dddacbb14981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 32, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_105899f5edd493c4eb71032a14b38afc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 11, 16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_48996548633eeb3d663bd81983882a21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([300, 256, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_886f76c8d62b5d29d0bba4ce699fc66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 300, 256], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b5f63cbadb0d36809db2d4093236555a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 8, 7, 8, 7, 96], dtype='int64').reshape([6]),
        ]


class PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91e6eff5aee7fe8e518e8dddae5b7bcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 64, 49, 3, 3, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_23e407b2600c8c5e2eb09a2ac7c0ac73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ff91a87f51b71fdd01632089828fcb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 3, 4, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_74aba368c58a586a959f54a95125bbe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 4, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 128], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_6762a2714cd5f0c75f0ff867b207dafb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 198, 3, 3, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_56732c3c1f7efbbdcffd1da7c62b273c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 3, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 198, 192], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_93422832bba00db134354cd6f3c2bc60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1960, 16, 2, 4, 6], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_b9d57106b54a41cf909fa8cb3fc8ceec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1960, 16, 4, 6], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c0156cfed6741fd3941c393340f4d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f80261241dca370c98381bc30b86e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8dccc8bc5773bc4065087b4e7bb99306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_87ac56114947f955f16db848380a15ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 784, 6, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_582c4d804520c04931648161479d67ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 192, 28, 28], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_69ae06f9483c5274260de6bbe41421ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 192, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2d9a363e54c4eb4a331f0c0fafb73e66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 49, 2, 6, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_5ce91c64d63fb5dde4a16f8b81cb4615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06460928171873093, 0.11852037161588669, 0.4668518006801605, 0.4338199198246002, 0.34104853868484497, 0.29185888171195984, 0.10113299638032913, 0.06567597389221191, 0.44941550493240356, 0.4201486110687256, 0.35819512605667114, 0.3011251389980316, 0.19713793694972992, 0.3007983863353729, 0.48699697852134705, 0.29727399349212646, 0.493547648191452, 0.2533818185329437, 0.0853106752038002, 0.46487218141555786, 0.165374293923378, 0.17134487628936768, 0.31911700963974, 0.39892691373825073, 0.05332564562559128, 0.4865640103816986, 0.058651529252529144], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_85e8663c4b36aacb75e92536f9bdb660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ae195ac5ac6948659191ec1413a04b06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2457106113433838, 0.14956028759479523, 0.272349089384079, 0.28523972630500793, 0.1871337890625, 0.3845573663711548, 0.4802970290184021, 0.3273279368877411, 0.2611807584762573, 0.4178078770637512, 0.03777754306793213, 0.23063205182552338, 0.0005907635786570609, 0.19496837258338928, 0.0834074541926384, 0.26149293780326843, 0.4555708169937134, 0.4083622694015503, 0.4904957115650177, 0.23973101377487183, 0.20747047662734985, 0.32015472650527954, 0.04469742253422737, 0.19461622834205627, 0.2533906102180481, 0.4906653165817261, 0.17441265285015106, 0.4651762843132019], dtype='float32').reshape([28]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c804d74f0649ca974979e2aaba321ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([258], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c804d74f0649ca974979e2aaba321ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([258], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c804d74f0649ca974979e2aaba321ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([258], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b7daf6956c5d6dbeb3ebe2088ca1f8d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22842250764369965, 0.2582748234272003, 0.07155562192201614, 0.4213123321533203], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2769849c301d993c9ce0602960cfe8ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32285594940185547], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2db835d49c10fb8e61337d3e3d97b160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.19056516885757446]], [[-0.6171947717666626]], [[-0.2424192726612091]], [[-0.1877579689025879]], [[-0.08308002352714539]], [[-0.11546275019645691]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_74dc619386d2c45dc0dff19044c4c2a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.2167588770389557]], [[-0.4417083263397217]], [[-0.4992922842502594]], [[-0.4906761348247528]], [[-0.34343820810317993]], [[-0.38006407022476196]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_732e6ef16bbca256a315f380bfd4172b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.124290943145752]], [[0.7009971141815186]], [[0.8767253160476685]], [[0.8192540407180786]], [[1.0621317625045776]], [[0.9383822679519653]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c2384b7f34845b6eb299ea7acaf6b3e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.8433278799057007]], [[0.6808311939239502]], [[0.9778209924697876]], [[0.958483099937439]], [[1.0988681316375732]], [[0.7455201745033264]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7dd58d27498d8bdd7b1b796cc2f3b60a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16, 32, 128, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f311db7c73898b572af69b4603d1ea93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cce4a4318d19a619b64f4fbedd51a8a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3332940936088562, 0.09769731760025024, 0.43830451369285583, 0.009228624403476715, 0.16223640739917755, 0.11507797986268997, 0.0893772691488266, 0.22552497684955597, 0.052476346492767334, 0.44484132528305054, 0.4949187636375427, 0.20712272822856903, 0.023928960785269737, 0.3156645894050598, 0.2349514365196228, 0.28845635056495667, 0.1664237380027771, 0.444826602935791, 0.3088386356830597, 0.19834139943122864, 0.32904279232025146, 0.3947077691555023, 0.4195897579193115, 0.014351009391248226, 0.4399889409542084, 0.44951435923576355, 0.47202175855636597, 0.3262561559677124, 0.39623090624809265, 0.07782911509275436], dtype='float32').reshape([30]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_badc10645ab162cad07f9d1f694a22b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11763664335012436, 0.19943761825561523, 0.2622498571872711, 0.31616735458374023, 0.001056869514286518, 0.06995536386966705, 0.18180325627326965, 0.38399800658226013, 0.15767301619052887, 0.39410117268562317, 0.4438955783843994, 0.2776387333869934, 0.0789950042963028, 0.1973235309123993, 0.3119330406188965, 0.3275054395198822, 0.4733019769191742, 0.4643199145793915, 0.4826446771621704, 0.3815286159515381, 0.008363288827240467, 0.01255305390805006, 0.18946005403995514, 0.29119038581848145, 0.01884123496711254, 0.14236989617347717, 0.16595062613487244], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2488a9e945280c1f612ea70a68fb7305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24339798092842102, 0.46689680218696594, 0.14140304923057556, 0.47299179434776306, 0.028293166309595108, 0.4629482924938202, 0.26495954394340515, 0.053110573440790176, 0.37322211265563965, 0.15704494714736938, 0.17712032794952393, 0.06270186603069305, 0.18813490867614746, 0.4491448700428009, 0.23343463242053986, 0.14770770072937012, 0.16615281999111176, 0.445487380027771, 0.4283473491668701, 0.33060166239738464, 0.14955562353134155, 0.11336975544691086, 0.013417585752904415, 0.06433072686195374, 0.3512799143791199, 0.15523624420166016, 0.44879573583602905], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e303cb45359764f66b219c4e31a7d5a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21433231234550476, 0.14448918402194977, 0.3197583854198456, 0.46643924713134766, 0.26216596364974976, 0.49535366892814636, 0.33114930987358093, 0.2132221758365631, 0.008679743856191635, 0.022425219416618347, 0.1886783093214035, 0.1404712051153183, 0.1191626712679863, 0.4659154713153839, 0.043324828147888184, 0.3108152151107788, 0.3062174320220947, 0.20970258116722107, 0.3221311569213867, 0.3665383756160736, 0.14851608872413635, 0.3041526973247528, 0.24670255184173584, 0.31109604239463806, 0.3691222071647644, 0.11542730033397675, 0.21138763427734375], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_28e8803dfae2bf58a6fec5d379984109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 8, 7, 8, 7, 96], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_f4913bdccc219ee4c1f8ccb9bd7cbd00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 64, 49, 3, 3, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_460474e6cc192be5b38136e7294590ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 3549, 4, 19], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b88559b8a5d2fb1ea10c2afd7857b31a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([115200, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_21741ffb65a59bfb9577b1c59185107e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([28800, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6067065086972d3b73b175d13fbab79f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([7200, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f3a5c7a5426bfacbfed1968f3188fd72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([1800, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1dff77c4da28ecbd4fb4317778ed81df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([450, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_969579ec3704db7415ccc9fc54572922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 240, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7d10e6e4c5f5c79711afacef5a2508e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 120, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_93e25f3af0bd522a82266c32c79453f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 60, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7a8f069a8c66270448418dd5ca4f4a6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 30, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_9823e2f67ef8239b4aec5a5d1986fe13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 10, 15, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_fc372e43a257bbb98727c7ad4c235103(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 240, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2b53f9221aef3ac1b4ad97948cbed312(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 120, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_150db303b9e578c52df3db7c8628be06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 60, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7d1a336b2819a7431d5736ad98fbfd7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 30, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_398bb04674e2c3735272a49e254349c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 10, 15, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b704e8a2057ee823f83fcb49cedfa20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_37dd8ade427fac57583faed4890971dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4ee3b807c30c71482b85c15453249e3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f10b91eeaa45ba5106bd12c92acf2d26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_940b65524433657086eff3256fa78c3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a671cacc3d31b6b49d8e71317c1b430d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([3136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 28, 28], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e5ec1e605f87a274c28649cf9bc487d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_242b271651ab2f42d495ee1d092f959f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.46424782276153564, 0.4705996513366699, 0.42634397745132446, 0.35580918192863464], [0.16565507650375366, 0.09719990193843842, 0.16619716584682465, 0.37486857175827026], [0.24447554349899292, 0.17953190207481384, 0.07921028137207031, 0.42684468626976013], [0.4760080575942993, 0.08616776019334793, 0.3810620605945587, 0.0039510601200163364], [0.25089192390441895, 0.4652281701564789, 0.40618160367012024, 0.3080861270427704], [0.32649412751197815, 0.18960219621658325, 0.2721419632434845, 0.15155068039894104]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([0, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c804d74f0649ca974979e2aaba321ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([258], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c804d74f0649ca974979e2aaba321ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([258], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c804d74f0649ca974979e2aaba321ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([258], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c804d74f0649ca974979e2aaba321ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([258], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c804d74f0649ca974979e2aaba321ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([258], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c804d74f0649ca974979e2aaba321ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([258], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_29fb9c39eda326b391a064e451c579e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([100, 256, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e442ecd66d5901639c355073b02dc02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 100, 256], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_89f9ac9c63e7ccab32fa26626fdea3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47607406973838806, 0.3472485840320587, 0.10741978138685226, 0.3268607556819916, 0.028444116935133934, 0.18968714773654938, 0.1720195859670639, 0.12753750383853912, 0.3384349048137665, 0.11674002557992935, 0.09715169668197632, 0.3919233977794647, 0.3752148449420929, 0.012128832750022411, 0.14214417338371277, 0.27708861231803894, 0.11623965948820114, 0.316640704870224], dtype='float32').reshape([18]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_07bcf3c5cb3e393a981bc805af40d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 18, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 2, 9, 112, 112], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_d9f30cc45fa3f45a500fb5c24f0baa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 288, 12544], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 2, 16, 9, 112, 112], dtype='int64').reshape([6]),
        ]


class PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a0d9baedcc1ef8c1f45d64a87cfab3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 16, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 32, 112, 112], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d96366ee78c558d8303ed12b3d81b83f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10371440649032593, 0.3718690872192383, 0.11184684932231903, 0.4691217243671417, 0.27857381105422974, 0.09794922918081284, 0.03927387297153473, 0.31190451979637146, 0.2813241481781006, 0.2726524770259857, 0.4290771484375, 0.46386635303497314, 0.4334547519683838, 0.16444368660449982, 0.38464412093162537, 0.32888707518577576, 0.3701514005661011, 0.007624500431120396, 0.4428855776786804, 0.3563711643218994, 0.3173627257347107, 0.4352709949016571, 0.17733845114707947, 0.35450300574302673, 0.4631364941596985, 0.34386664628982544, 0.4727720320224762, 0.1618299037218094, 0.08435483276844025, 0.4169885516166687], dtype='float32').reshape([30]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_db00518310debd943fe025fa2ea5d14e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4907311797142029, 0.2802000045776367, 0.3868299722671509, 0.20437335968017578], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_38707d52d96627022272ec69125a7276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.41381654143333435], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9f6e969dac0bf14f4cb5211432b344fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0006107513909228146, 0.4980224668979645, 0.09194213151931763, 0.24664801359176636, 0.003547533182427287, 0.22427155077457428, 0.14103838801383972, 0.43646109104156494, 0.18649671971797943, 0.4875296950340271, 0.4018765985965729, 0.31609025597572327, 0.36809781193733215, 0.3689664602279663, 0.01992814429104328, 0.30289915204048157, 0.19099527597427368, 0.2533676028251648, 0.072096087038517, 0.3927765488624573, 0.4612211585044861, 0.1182871013879776, 0.3730167746543884, 0.4420868754386902, 0.4957417845726013, 0.10735879838466644, 0.24171218276023865], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d8c3ad013e5e88f8095e74d509cdf926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_065a37379efff1e978aab35bd4f5e4d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.027834607288241386, 0.08033515512943268, 0.1517733782529831, 0.47431546449661255, 0.06370123475790024], dtype='float32').reshape([5]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_96d97133472ab515bab978e3095935c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18604914844036102, 0.3080838918685913, 0.2649553418159485, 0.26149383187294006, 0.4871636927127838, 0.1859550029039383, 0.38509342074394226, 0.34411531686782837, 0.44187912344932556, 0.4060733914375305, 0.12789003551006317, 0.4741241931915283, 0.12385957688093185, 0.19852232933044434, 0.04140293970704079, 0.35921013355255127, 0.25158265233039856, 0.24189026653766632, 0.08602314442396164, 0.2212895154953003], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_aa40e7663f5f9cf9eb373012d1801f24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17142826318740845, 0.2332010120153427, 0.13102346658706665, 0.09057428687810898, 0.3809174597263336, 0.31674641370773315, 0.1272394061088562, 0.436483770608902, 0.48855432868003845, 0.36683595180511475], dtype='float32').reshape([10]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c0156cfed6741fd3941c393340f4d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_db1388a65a19770df872c92d63ea55d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 20, 128, 256], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_ca82a4da28970b48bd75e52ae8bfebe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 2, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 40, 128, 256], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1e2c7f966d81db07ffdd7e160cd0181b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 40, 64, 128], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_8c2e63af0d0bee98a4f0c30d350a56bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 2, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 80, 64, 128], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e5ec1e605f87a274c28649cf9bc487d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4308d1db7b8ccffd990c675375308557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4835236668586731, 0.18737399578094482, 0.4939134120941162, 0.0895947590470314, 0.026810066774487495, 0.1268116682767868, 0.37494948506355286, 0.04894789308309555, 0.3902507424354553, 0.44619643688201904, 0.1254793256521225, 0.49778127670288086, 0.013724558986723423, 0.12982036173343658, 0.4037642776966095, 0.0820649042725563, 0.12104443460702896, 0.22918403148651123, 0.08095712959766388, 0.3118191659450531, 0.23261098563671112, 0.1339697688817978, 0.2959449291229248, 0.40542304515838623], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class PrimitiveOp_5648fb57a7fc0f60bdde9ea58affba12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc776162284325971068392cf6a41540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5648fb57a7fc0f60bdde9ea58affba12
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2100], dtype='int32'),
            paddle.to_tensor([1, 2100], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3336f42b47b1bee50d05c21effdc029d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2537922263145447, 0.22381779551506042, 0.43471693992614746, 0.11890356242656708]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_459fac38a50cd552ee2db8e229bcad55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2100, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f9d9d0167a52a874768c07728a6fc2d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 152, 272, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 128], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3356c99b8d94f62261f3267d1b9900bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_dbec0c5ad84ff6f21f03903ca8d46775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 196, 12, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f561dfce2dc5f61884817f6dedcfa452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 384, 14, 14], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a94745db582e62919bf52edec9b9ce96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 384, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_1c009caf4435d3ba6206f215d613303e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 49, 2, 12, 32], dtype='int64').reshape([5]),
        ]


class PrimitiveOp_8c7ce804f9b0f6be3beb30396abf2248(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29350666b17d3127f4e6f1d1ed973533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c7ce804f9b0f6be3beb30396abf2248
    def get_inputs(self):
        return [
            paddle.to_tensor([128], dtype='int64').reshape([1]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_a46d0996b6c4ea0d975b1f75a72eb1b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c7ce804f9b0f6be3beb30396abf2248
    def get_inputs(self):
        return [
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_2be732df023c2a345cad67a406ccf0b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c7ce804f9b0f6be3beb30396abf2248
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_62606c67b7f49236c76678b9c0779e2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 16, 8, 128], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b4d48f961bdd6507ed527888faea93b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a16c788d5410df33cd8f05119198cbf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47866010665893555, 0.07238171249628067, 0.3800481855869293, 0.47888174653053284], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_38eb55c3270e4aa68aac22df06782176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42756983637809753], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4ddd6a72ef87c08de482162965da05cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03537175804376602, 0.3485872149467468, 0.2182174026966095, 0.22238129377365112, 0.44054776430130005, 0.38505855202674866, 0.19396695494651794, 0.29349374771118164, 0.14737622439861298, 0.03577267751097679], dtype='float32').reshape([10]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d9287e55ad601d146d9a6b779c009d8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([96, 96, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bda57a1a3a805afbc645a6cf5b9f5be7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([48, 48, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dc274caad42188845dab9730014bf96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([24, 24, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d9287e55ad601d146d9a6b779c009d8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([96, 96, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bda57a1a3a805afbc645a6cf5b9f5be7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([48, 48, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dc274caad42188845dab9730014bf96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([24, 24, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2de81fcc00d7ffa4a406ea4659814fd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 7, 7, 768], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a4fe737c29f2de0610de8192d9e1d805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ba4b6b667dd2b1c4aa1f3286832036b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4897165298461914, 0.19547180831432343, 0.31587594747543335, 0.4530157148838043, 0.39336615800857544, 0.265292763710022, 0.10776548087596893, 0.20151233673095703, 0.159870445728302, 0.06994098424911499, 0.43550464510917664, 0.152782142162323, 0.2769245207309723, 0.4957667291164398, 0.06416304409503937, 0.4347667992115021, 0.3767150640487671, 0.24708451330661774, 0.16153693199157715, 0.24565987288951874, 0.3230544328689575, 0.12164636701345444, 0.4440462291240692, 0.18994607031345367, 0.34309229254722595, 0.330647349357605, 0.11721905320882797, 0.058021821081638336], dtype='float32').reshape([28]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b1921d7d8628c473d7f241e9ee8b7739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09685768187046051, 0.4633624255657196, 0.3370261788368225, 0.2218378782272339, 0.3003818690776825, 0.3977978527545929, 0.22157736122608185, 0.34453585743904114, 0.01729939877986908, 0.2940759062767029, 0.00028275715885683894, 0.428407222032547, 0.45860815048217773, 0.0640517920255661, 0.15221236646175385, 0.41897520422935486, 0.36749085783958435, 0.35694530606269836], dtype='float32').reshape([18]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e5ec1e605f87a274c28649cf9bc487d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3c79180c333f7e6f466d0ce1dfb75d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23480238020420074, 0.43731117248535156, 0.10146879404783249, 0.17813141644001007, 0.0595996119081974, 0.3097524642944336, 0.3295620083808899, 0.497146874666214, 0.23725514113903046, 0.4181806743144989, 0.005523432046175003, 0.041548725217580795, 0.3885549306869507, 0.48227620124816895, 0.3439391255378723, 0.1416143923997879, 0.3807408809661865, 0.20559805631637573, 0.15291662514209747, 0.4659992456436157, 0.2486792802810669, 0.14446760714054108, 0.3553442656993866, 0.11915403604507446, 0.28219401836395264, 0.48845967650413513, 0.3902752101421356], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d8c3ad013e5e88f8095e74d509cdf926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0bcb54e425e39eb1d9d6662781dac837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.019995983690023422, 0.46340736746788025, 0.33318284153938293, 0.37921011447906494, 0.17319579422473907, 0.39732056856155396, 0.30064859986305237, 0.14454451203346252, 0.1661507934331894, 0.33594393730163574], dtype='float32').reshape([10]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_232589db835d21d5d459f2c6dc67aad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 512, 1024], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cd664a662fbbea631f3d56c19bbd54c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_40f112bbf4fdfa99815543a3407fe62c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9812b044f1991dc5c83595d8f4c47a01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 86970, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_64b1689a815442f3429115a9b6638c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 86970, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fe4047ddb04f167766072ae3fe85a27f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.16287018358707428, 0.442844033241272, 0.1370561569929123, 0.15503418445587158, 0.4858059883117676, 0.26686617732048035, 0.3084322214126587, 0.2806321382522583, 0.37827152013778687, 0.20173630118370056, 0.3930991590023041, 0.06881558150053024, 0.23861578106880188, 0.25531554222106934, 0.0681443139910698, 0.24908150732517242, 0.18759994208812714, 0.419292151927948, 0.01486904639750719, 0.12260129302740097, 0.4606020152568817, 0.15763233602046967, 0.11852873861789703, 0.058420710265636444], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_410caaedc4bb4330dee09f594a7b942a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c7ce804f9b0f6be3beb30396abf2248
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_dffd2d6b8fc44d301bfbd07e0c1b358b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08989069610834122, 0.16152428090572357, 0.12029926478862762, 0.056862469762563705], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_63a63f64f9fa0597fa8b5a080b0a350c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12036552280187607], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e51c8625c57e048ee87004234d645052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 216, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 128], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_660148c1306fc74c6de11fe60506747e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([43136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7f79598f7a510bddfaf88c7476e394e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05286628007888794, 0.241181418299675, 0.3427754044532776, 0.47604626417160034, 0.11741075664758682, 0.14756019413471222, 0.022136956453323364, 0.2565590739250183, 0.4010572135448456, 0.12779149413108826, 0.4315584599971771, 0.35410240292549133, 0.20665931701660156, 0.20922577381134033, 0.23823784291744232, 0.07190393656492233, 0.15548914670944214, 0.2402566522359848, 0.10341783612966537, 0.3127906620502472, 0.0025843135081231594, 0.007317973300814629, 0.22064191102981567, 0.4804287552833557, 0.19357265532016754, 0.2590285539627075, 0.21999569237232208], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_85e8663c4b36aacb75e92536f9bdb660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d40005bbd5a31c482e1d147ae2c0b25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3862338662147522, 0.13770005106925964, 0.43296223878860474, 0.0528138168156147], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cd02fc59121b56671df83323dc146f16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33504700660705566, 0.21198640763759613, 0.048459380865097046, 0.46090948581695557, 0.30291855335235596, 0.05838097259402275, 0.3479037880897522, 0.35365232825279236, 0.3773714303970337, 0.45497995615005493, 0.27654698491096497, 0.4180799722671509, 0.35187312960624695, 0.39763543009757996, 0.4553602337837219, 0.20647253096103668], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_17012acf89d4f8735b0375c837802488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([78], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_17012acf89d4f8735b0375c837802488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([78], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_17012acf89d4f8735b0375c837802488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([78], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_967615fd70a15c121ddb26c7f07bd2c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14210569858551025, 0.3227754235267639, 0.32499727606773376, 0.4789038598537445, 0.17756974697113037, 0.0901087149977684, 0.23322822153568268, 0.051212336868047714, 0.3015468418598175, 0.3246433138847351, 0.32804813981056213], dtype='float32').reshape([11]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e458380f2fe6841336df0e6b58b9fd80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6e29ab266cb41434bc9c1514e3ec3007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 242991, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c28b028f0f3868e29430ea90f597ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 242991, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c0156cfed6741fd3941c393340f4d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f80261241dca370c98381bc30b86e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c6ea581b0f7ecbdf4674845fb1dd4d80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([64, 512, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 8, 8, 128, 13, 13], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_388039d77bff77fe4c98ea3ca2faf9a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([4, 13, 13, 128, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 512, 8, 8], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_226db9080065987f256ec5ee3bdeab83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_62515a67db0f2c7d3ef7a1703154077b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 196, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 4, 49, 56, 56], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_1668b8e15e871c0aba4dfa802d2b8fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 3136, 3136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 4, 16, 49, 56, 56], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_493fefa28174b91de855e6c95303aa1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 16, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 64, 56, 56], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e5ec1e605f87a274c28649cf9bc487d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0b1acb06b0664b5c49b3197f81a13f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d5b7a41f24767050c4fbf65dd0c8de50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44761428236961365, 0.43118372559547424, 0.22728057205677032, 0.23422099649906158, 0.3976551294326782, 0.21660906076431274, 0.0913834422826767, 0.2451097071170807, 0.34926435351371765, 0.2152893841266632, 0.34884628653526306, 0.10444789379835129, 0.40315836668014526, 0.43056315183639526, 0.21740159392356873, 0.1747661978006363, 0.02957768552005291, 0.3052937090396881, 0.49654561281204224, 0.38624319434165955, 0.4902457892894745, 0.22382493317127228, 0.06349996477365494, 0.33312398195266724, 0.3866255283355713, 0.17645680904388428, 0.00967120099812746, 0.1487502008676529, 0.438001424074173, 0.02855030819773674], dtype='float32').reshape([30]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f311db7c73898b572af69b4603d1ea93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a903f6e3d93c876efae555eb24f92f05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 3136, 3, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f4d53dacf8653c48ec383e7bf661a441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 3136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 96, 56, 56], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5cc4bd0aaefb368af16632871a678272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 96, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e0db8ef02f983c7d5e0eb847f71efa25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 49, 2, 3, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7ef6c51dd626822425a0a7a84bc8b703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 3, 4, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_18884e65102af273b2f5c32e2b945d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 4, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 128], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_678eef615c912d53f7af479624724855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_64526bea52e2754646ee289fe799cc4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39255741238594055, 0.20061296224594116, 0.4770492911338806, 0.24852369725704193, 0.08301194757223129, 0.08201950788497925, 0.381465345621109, 0.20208585262298584, 0.27806535363197327, 0.1560158133506775, 0.3637750446796417, 0.19218532741069794, 0.0041387006640434265, 0.44846808910369873, 0.31852248311042786], dtype='float32').reshape([15]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_418b966dd4b7878f866e2e0d3474eef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4187091886997223, 0.4834619462490082, 0.467659592628479, 0.2921316623687744], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f5a364ba3b820bb68c7e28d61ceb1366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.235874742269516], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_160c558475a13841b8414024af3d4ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 220968, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_21d03d4478eead64caaa90c2c964cccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 220968, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_007c3d1612caa5831dad7b8c8c6b9e0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.005234016105532646, 0.4531991481781006, 0.3853684067726135, 0.31216976046562195, 0.22510886192321777, 0.3198269009590149, 0.42550066113471985, 0.08712606132030487, 0.3956355154514313, 0.25680696964263916, 0.23739232122898102, 0.3796047866344452, 0.34644004702568054, 0.04048977047204971, 0.06147902458906174, 0.2101835012435913], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9ba8bbd48999d1f02a23cfd8def757fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24054042994976044, 0.10996941477060318, 0.45132285356521606, 0.49869784712791443, 0.28534162044525146, 0.27436405420303345, 0.24714595079421997, 0.23148241639137268, 0.21625499427318573, 0.012740080244839191, 0.21534620225429535, 0.34772148728370667, 0.4850476384162903, 0.24668489396572113, 0.4515797793865204, 0.17637042701244354, 0.4703393876552582, 0.40507426857948303, 0.1684473305940628, 0.18838092684745789, 0.1915687769651413, 0.06606480479240417, 0.16552570462226868, 0.14468005299568176, 0.2090993970632553, 0.2713947892189026, 0.35166189074516296], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f7f62f2f0f020f9e6a466633eb6ff475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_40f112bbf4fdfa99815543a3407fe62c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3e484e500c365f33aa9ccf6cfd4f560a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 512, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_ae3942d02c9b2fd91183f13f1759fca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 19, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a4fe737c29f2de0610de8192d9e1d805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7ddf83683011fb5c973dc8a9960b83ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 7581, 4, 17], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_844d6171db47c440f613d91a6c9d6b21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([528, 96, 4, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 96, 96], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_15616f0dc08a1db431f253cce2bcdd98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([528, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 1, 24, 48, 2, 96], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_69d17f3f992d3236e483bfbf374999fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 48, 24, 2, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 48, 48, 96], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_de7131613f634fcea1341a7f51db7a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34376952052116394, 0.10028111934661865, 0.4983377158641815, 0.4409283995628357], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_52f9f8d8f8ca7921883d87838609845e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.032395415008068085], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8dccc8bc5773bc4065087b4e7bb99306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e61553404730445cef67b20869cfe107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 1, 3, 3], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_558afb6e38b452d9e3692126e3404b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15359817445278168, 0.12305621057748795, 0.29325833916664124, 0.250827819108963], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a2641a691802ddc2088dffd5674e44f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26530763506889343], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2b6f339bdc99694eb513483c6387e720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 4, 7, 4, 7, 192], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_499404e4daabc3b6f9a0e17f0e4041c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 16, 49, 3, 6, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_4bd8864870f8c8148e3748d49d6872eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 512, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b4c54361e7aa78512061c230728f1424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 21, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3c0c80f30e78e0806a7534c485819e67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([12, 288, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([6, 2, 1, 12, 24, 192], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_c3330ed7e29b339bb9facd519b3d8979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([6, 2, 12, 1, 24, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([6, 24, 24, 192], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ec38754fb44d8dc4f9dc454b0a94e3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 16, 64, 128], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b4d48f961bdd6507ed527888faea93b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6de5b16766458e0da3298c82fb19a42e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4725, 4, 17], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b87616b3a85ddc477c5f7e2756797078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10133931040763855, 0.282049298286438, 0.1450863629579544, 0.293549120426178], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5b0051e637fb2ed24d858defdbf2ae0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22334277629852295], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0f4b8ead002dc36a70bfccb1bca96ede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 16, 64, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f80261241dca370c98381bc30b86e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9b9aa787b686ad7a49ec92e73315d042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18656504154205322, 0.03847033530473709, 0.2931731641292572, 0.30716535449028015, 0.3304098844528198, 0.47976255416870117, 0.28181272745132446, 0.1657336801290512, 0.09241204708814621, 0.07179971039295197, 0.21914009749889374, 0.2036820948123932, 0.4740002155303955, 0.34855908155441284, 0.36259663105010986, 0.052021343261003494, 0.1680237352848053, 0.05093527212738991, 0.4821746051311493, 0.0844661295413971, 0.19476205110549927, 0.27226924896240234, 0.380504310131073, 0.3280499577522278], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bb134a21097aabb9d0fca1f1b78b67ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26143741607666016, 0.31354454159736633, 0.3454189896583557, 0.346386194229126, 0.024124952033162117, 0.1401016265153885, 0.43856361508369446, 0.12268566340208054, 0.4728694558143616, 0.047065261751413345, 0.05490707606077194, 0.06764058023691177, 0.12034686654806137, 0.46121907234191895, 0.36339622735977173, 0.15023884177207947, 0.20135626196861267, 0.03405923768877983, 0.01789616048336029, 0.04741783067584038, 0.48182329535484314, 0.39881134033203125, 0.48119792342185974, 0.06364081054925919], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b244a4784ac3469718c61c4d188474d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.019333021715283394, 0.25483834743499756, 0.447208046913147, 0.1557384580373764, 0.0626818984746933, 0.14274196326732635, 0.4807971715927124, 0.21238580346107483, 0.4113515317440033, 0.4200870096683502, 0.04387315735220909, 0.012349985539913177, 0.35682716965675354, 0.3714551031589508, 0.2868078052997589, 0.4088013470172882, 0.4972761273384094, 0.20539624989032745, 0.018960798159241676, 0.28467515110969543, 0.08199106156826019, 0.04765501618385315, 0.12095426023006439, 0.11579488962888718], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_dd8ec4ea39a5ca8ed8f7be6d033cecdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.16443945467472076, 0.4716094136238098, 0.12132081389427185, 0.02427920699119568], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2a1dff215168a92408165f9278dbe80c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19837483763694763], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a17d2dc43aa9fb4d6aee608dc88aa5f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 153450, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bcce97a6b60d422c230affecf1d3977f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 153450, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ad80dfbc68ad9c7ad0709f5ab3323ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 12, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 64, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0f838567775c6f5e20707a83c51a669d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 144, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f311db7c73898b572af69b4603d1ea93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2a5b3ec6cb426344069c78ba43eee98c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([2352], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 28, 28], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_cb5b98cbe4f7d629cdaabe8db2445e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 2304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 577, 3, 12, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_3688aa8103c0ad56f0c34a8f3e72f631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 12, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 577, 768], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8e6f5c68b7c7f2a1d5b5cb74c10c9013(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2241799384355545, 0.14815174043178558, 0.15414701402187347, 0.48750820755958557, 0.06894189864397049, 0.4504745602607727, 0.25748685002326965, 0.4988856017589569, 0.32149583101272583, 0.08806567639112473, 0.15711341798305511, 0.018915262073278427, 0.42349332571029663, 0.18404455482959747, 0.19405212998390198, 0.20503471791744232, 0.03463481366634369, 0.1507311910390854, 0.1342015564441681, 0.4745321273803711, 0.4136272370815277, 0.49665167927742004, 0.3469334840774536, 0.3201349675655365, 0.33642908930778503, 0.04985976591706276, 0.3372310400009155, 0.420236736536026], dtype='float32').reshape([28]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_918e13c281a13bd80608d45c9e0ac990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 14, 14, 384], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ddaba0f279687a398875094bfd5282bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 64, 16, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8610f15fe9ba9fc7c73abc5a943b9cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3764646053314209, 0.43379876017570496, 0.39359864592552185, 0.23601405322551727, 0.3794044852256775, 0.1119103729724884, 0.3893990218639374, 0.01082636695355177, 0.2554991841316223, 0.29726508259773254, 0.2694675326347351, 0.37152862548828125, 0.442438542842865, 0.2681443989276886, 0.20980069041252136, 0.1441047489643097, 0.03168831765651703, 0.19934070110321045, 0.3154596984386444, 0.425428181886673, 0.05098959058523178, 0.32411813735961914, 0.15539522469043732, 0.29909712076187134, 0.2877471148967743, 0.1100429892539978, 0.19862845540046692, 0.2084355503320694, 0.0038356545846909285, 0.45225006341934204], dtype='float32').reshape([30]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c79bb9e4b03c3307960b7bb05bf50ea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 197, 2, 6, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_dcbe90ef6f0f3233f5496f9ee2565c91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 197, 6, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_89d268d9fad16b87212f2957b239908c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24803392589092255, 0.30121737718582153, 0.01479188259691, 0.09637084603309631, 0.060502372682094574, 0.06495889276266098, 0.14499694108963013, 0.356498122215271, 0.24755334854125977, 0.0047884369269013405, 0.35378122329711914, 0.036284931004047394, 0.14451508224010468, 0.4248710870742798, 0.22819094359874725], dtype='float32').reshape([15]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b4d48f961bdd6507ed527888faea93b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ae61f92f681fdc7d0cbd4113a84d3bfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3601657748222351, 0.10146880894899368, 0.06412724405527115, 0.27262794971466064, 0.16660180687904358, 0.36374256014823914, 0.14419548213481903, 0.0434267520904541, 0.19956403970718384, 0.331539124250412, 0.05061684921383858, 0.12478219717741013, 0.031546905636787415, 0.17220327258110046, 0.035069625824689865, 0.049209415912628174, 0.02681557647883892, 0.4352574348449707, 0.30887383222579956, 0.3422398865222931, 0.4375320374965668, 0.15565863251686096, 0.3129275143146515, 0.05733947083353996, 0.3958487808704376], dtype='float32').reshape([25]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_67885294bc04c397eaa071255384cd0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_95ff496e52a40d31d3f34e9cee03498e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([384, 96, 2, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 96, 48], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3a8e8a132fd11c03b729fa0ef426ddd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([384, 96, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 1, 96, 96, 1, 48], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_f0753404abc070c20e5f8920b4eebf6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 96, 96, 1, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 96, 96, 48], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a3ab1719b3f7d0fe3bf959e53457e952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 28, 28, 192], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0755dfd9a6c93b7f2ea09b68a399eb9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 4, 7, 4, 7, 192], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_58010b0bce08203f99bde903e68d7c27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 16, 49, 3, 6, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_c15e467642c118e37c9fee3a9d6d2772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 16384, 2, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6e2fd4818d816d2610240085a5bbf457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 64, 256], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_830c557d255319231c6fb806ba9955d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2c1a47840686a79937a9275812c4008e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 2, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_87b6e52e378353bf1746a038fa3b39e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 2, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 16384, 128], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_0db984247e5628b1672473b6a6f8de0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2706141471862793, 0.3993256986141205, 0.2517136335372925, 0.24301311373710632, 0.2843993306159973, 0.49756133556365967, 0.25193703174591064, 0.36523500084877014, 0.4463474154472351, 0.34952881932258606, 0.3208235800266266, 0.4321683943271637, 0.48298102617263794, 0.4144379496574402, 0.36941608786582947, 0.4982113838195801, 0.23414961993694305, 0.32596147060394287, 0.16842985153198242, 0.0642484501004219, 0.310127317905426, 0.1135687455534935, 0.4742540121078491, 0.3582591116428375, 0.028468556702136993, 0.3446023464202881, 0.4451541602611542], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f7f62f2f0f020f9e6a466633eb6ff475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_77ac393481c49983439c91e41de95abf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47154712677001953, 0.08846384286880493, 0.4806261956691742, 0.3505251407623291, 0.43760088086128235, 0.08297809213399887, 0.2936021387577057, 0.2187899947166443, 0.0979703813791275, 0.019358061254024506, 0.14587779343128204, 0.41008466482162476, 0.2182627022266388, 0.2402622401714325, 0.3501252830028534], dtype='float32').reshape([15]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a903f6e3d93c876efae555eb24f92f05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 3136, 3, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f4d53dacf8653c48ec383e7bf661a441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 3136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 96, 56, 56], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5cc4bd0aaefb368af16632871a678272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 96, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e0db8ef02f983c7d5e0eb847f71efa25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 49, 2, 3, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_27155d9de9907a8d3ea6e0d1a3663568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16, 32, 64, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_641c71185607ef2624390245874ea8ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 185691, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c247b5238e129ce3ed48137feb2721e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 185691, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e5ec1e605f87a274c28649cf9bc487d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f60a11f5c597e603484fc5b6091453fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([7148], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f60a11f5c597e603484fc5b6091453fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([7148], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fc4d2607cd0c5d40ba3623dbfd4ed9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([121516], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4, 17], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f60a11f5c597e603484fc5b6091453fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([7148], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_63b8b5a5027f0cbb5fc7ecc55842c569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 49, 24, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_251afd04f8bc95c955f8aa07b5f7615d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1536], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 49, 2, 24, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a844bf92052d07143a38df0e35b6f0bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 8, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 64, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c6b6f3c7f6a340f137861a6d03ada98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 64, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8cebe312d7a2e58c5d1e0a555f2c2c58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 8400, 4, 17], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6801ce5379b40a26779eed291cec409e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.029256418347358704, 0.06164555996656418, 0.2207813858985901, 0.11275386810302734], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_293c0017f1eafca2d59bf342419ad1b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22667348384857178], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f311db7c73898b572af69b4603d1ea93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_226db9080065987f256ec5ee3bdeab83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f05a43ace285b4d4ac1c1553c13ee280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 4, 49, 56, 56], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_27b2301fbdbcc6ce648103c9e5acc184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 3136, 3136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 4, 16, 49, 56, 56], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_26dbcd4bd9f7fcb540640b851d4a90bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 16, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 64, 56, 56], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_945fca59373e41f913991129f8a1c012(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 56, 56, 96], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b5697afc8ee32bb18cf045f4eddd2f9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0aed96a9cbe8266a2045ac59af357aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4f9a839c307a1ac7108b07cfbf251d05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([16, 16, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b5697afc8ee32bb18cf045f4eddd2f9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0aed96a9cbe8266a2045ac59af357aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4f9a839c307a1ac7108b07cfbf251d05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([16, 16, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ca5aa789752502dd1c15f4a41a14d367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_729c1698bef6d05cb12e5cde0f988c23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.46713727712631226, 0.4755287766456604, 0.28019455075263977, 0.4221588671207428, 0.3353338837623596, 0.30886349081993103, 0.3970082402229309, 0.406235009431839, 0.3186098635196686, 0.1592205911874771, 0.18720930814743042, 0.26098331809043884, 0.003977949731051922, 0.16265875101089478, 0.31148943305015564, 0.46435844898223877, 0.4776941239833832, 0.19196023046970367, 0.16347691416740417, 0.23232167959213257], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0754f1918b9f53039d9d0cf1f9d27b56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 32, 64, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f80261241dca370c98381bc30b86e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f311db7c73898b572af69b4603d1ea93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b4d48f961bdd6507ed527888faea93b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_232589db835d21d5d459f2c6dc67aad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 512, 1024], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f2f427fb6de6dc2e474a9a043f5cd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11812976747751236, 0.4569990634918213, 0.32262200117111206, 0.4539250135421753, 0.46161723136901855, 0.027962252497673035, 0.49029865860939026, 0.0346745140850544, 0.48938167095184326, 0.463195264339447, 0.20857945084571838, 0.43197932839393616, 0.02523907832801342, 0.19653639197349548, 0.3426797091960907, 0.43827807903289795, 0.1322893500328064, 0.1832817941904068, 0.2959182560443878, 0.33154281973838806, 0.4549027979373932, 0.2605767250061035, 0.191399484872818, 0.3397514224052429, 0.07551240175962448, 0.395625501871109, 0.11812958121299744], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_85e8663c4b36aacb75e92536f9bdb660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6cdf279897a49a02711912e69120a41d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 3549, 4, 17], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8dccc8bc5773bc4065087b4e7bb99306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f311db7c73898b572af69b4603d1ea93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8b3dd4543837195c57bf8d0caf6ce4f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07998008280992508, 0.3795814514160156, 0.38602548837661743, 0.3560033142566681, 0.07171515375375748, 0.13767121732234955, 0.3039019703865051, 0.3747491240501404, 0.39127182960510254, 0.03411569073796272, 0.03493097051978111, 0.1621340960264206, 0.4014947712421417, 0.07974103838205338, 0.3354065418243408, 0.2860746383666992, 0.08680779486894608, 0.0888885036110878, 0.2877255976200104, 0.04290467128157616, 0.24113842844963074, 0.05063210800290108, 0.36901599168777466, 0.16386918723583221], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_671f63c084b1e249d2d3bab7a25a6e2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25431329011917114, 0.4427931010723114, 0.08209545165300369, 0.013483288697898388, 0.3257392644882202, 0.21759416162967682, 0.20277784764766693, 0.15045374631881714, 0.4908980131149292, 0.01286396849900484, 0.09119268506765366, 0.12351981550455093, 0.3320886492729187, 0.08019892126321793, 0.39889994263648987, 0.28920936584472656, 0.3824378252029419, 0.09345890581607819, 0.083846315741539, 0.004407353233546019, 0.4211543798446655, 0.10278724879026413, 0.18578791618347168, 0.3020758032798767], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d3efa2648b8733af5ac95d8d80c2964b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4088374972343445, 0.12412840127944946, 0.30896273255348206, 0.14724020659923553, 0.3771897554397583, 0.176796093583107, 0.1292911320924759, 0.3156766891479492, 0.34514325857162476, 0.29678407311439514, 0.1307784765958786, 0.0036876569502055645, 0.38948553800582886, 0.060126543045043945, 0.052999816834926605, 0.22186602652072906, 0.014729020185768604, 0.10389907658100128, 0.21261249482631683, 0.37377500534057617, 0.25414732098579407, 0.4316635727882385, 0.13645632565021515, 0.4335049092769623], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7e1e23f4db43ba76dabfb742aba14fe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60800], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 96, 200, 304], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f86218ef06d615256e9950a15eb8dc09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 113061, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db695e44566d2a5f48dbfcfbae00cd36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 113061, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9fc2486e054848a2b9e4386eb7d8a62b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 3, 2, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_fe57ed5e374a613856f584e84b13aed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 2, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 64], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_22f2b5534161bf7f4fb61442a3e33098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.008231737650930882, 0.1809747815132141, 0.4941970407962799, 0.1560303419828415, 0.26858294010162354, 0.40374988317489624, 0.4720913767814636, 0.48905226588249207, 0.40950676798820496, 0.34362295269966125, 0.04772471264004707, 0.4981529116630554, 0.43359610438346863, 0.2601719796657562, 0.07353013753890991, 0.3069276809692383, 0.3099937438964844, 0.1206049844622612], dtype='float32').reshape([18]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_596df5691b8bbd59f4f7a87db2540f7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12251222133636475, 0.031841810792684555, 0.32416704297065735, 0.16783101856708527], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2cd52dbf36fe6b9eeb3d75b01fe2fe4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.2886476218700409]], [[-0.29593008756637573]], [[-0.471481591463089]], [[-0.2454368770122528]], [[-0.17498314380645752]], [[-0.2884552478790283]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2a307f2b712515214564358e74453a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.554870069026947]], [[-0.269832581281662]], [[-0.6464014053344727]], [[-0.5651516318321228]], [[-0.20629549026489258]], [[-0.25149136781692505]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ac0da5c76a364c14653a7147fc76441c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.8499768972396851]], [[0.8985822796821594]], [[0.5331311225891113]], [[0.9018176794052124]], [[1.0358202457427979]], [[1.1671369075775146]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b52163e9ae17966866feed18ba4ca74a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.8478097319602966]], [[1.0442464351654053]], [[0.6927614212036133]], [[0.8982813954353333]], [[0.8514236211776733]], [[0.9219316840171814]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6506894a7036faaaa893169031bcc83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10237093269824982, 0.10831579566001892, 0.21228450536727905, 0.49737173318862915, 0.3437308967113495, 0.19317017495632172, 0.38485488295555115, 0.28498631715774536, 0.44435933232307434, 0.34744393825531006, 0.32678523659706116, 0.20103785395622253, 0.26013484597206116, 0.3101103603839874, 0.1776791363954544, 0.4299040138721466, 0.19602477550506592, 0.07021231949329376, 0.3518791198730469, 0.44830334186553955, 0.20475895702838898, 0.40439826250076294, 0.2781480848789215, 0.40839096903800964, 0.060484543442726135, 0.39760905504226685, 0.0025989445857703686], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_85e8663c4b36aacb75e92536f9bdb660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_23e407b2600c8c5e2eb09a2ac7c0ac73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bcf3a9fb639aed9064eea3393a54d299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44083404541015625, 0.049369122833013535], dtype='float32').reshape([2]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_49b2c395a6dc3a8589a5312762c40543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08215106278657913, 0.3222596049308777], dtype='float32').reshape([2]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ad3e9d95ed30ea09f441b53eba70d963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4568363130092621], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_35771dec8cf76bab437da9efa002b752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 198, 3, 3, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_531bc1334a813917cf572d405b4867dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 3, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 198, 192], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8dccc8bc5773bc4065087b4e7bb99306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_23e407b2600c8c5e2eb09a2ac7c0ac73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_097f1687535cbfba814c72c513d83616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13249199092388153, 0.28703057765960693, 0.027589386329054832, 0.013817519880831242, 0.41974806785583496, 0.33357805013656616, 0.2729302942752838, 0.3920206129550934, 0.3717678189277649, 0.0760609358549118, 0.18443796038627625, 0.06258518993854523, 0.13108012080192566, 0.1787378489971161, 0.47805994749069214, 0.3351810574531555, 0.2942843735218048, 0.2948574721813202, 0.156121164560318, 0.3806123435497284], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_fbf1da75da889e3a0e38e2dc348b8c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([22096], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fbf1da75da889e3a0e38e2dc348b8c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([22096], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cb492be7f48c7b7b303233ce395bde40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([375632], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4, 17], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_fbf1da75da889e3a0e38e2dc348b8c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([22096], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a1054949a2f2c4ea2574af8a5450797f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.051778003573417664, 0.15853163599967957, 0.184007465839386, 0.4035968482494354, 0.08098015189170837, 0.21231301128864288, 0.050704728811979294, 0.34025558829307556, 0.22989879548549652, 0.002397873904556036, 0.3064514994621277, 0.17127352952957153, 0.11324314773082733, 0.009763669222593307, 0.006200528237968683, 0.1323000192642212, 0.17113305628299713, 0.4427878260612488, 0.38522982597351074, 0.3787544071674347, 0.4689144194126129, 0.17210710048675537, 0.38978123664855957, 0.49802538752555847, 0.30641117691993713, 0.3758094906806946, 0.09675078839063644], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f7f62f2f0f020f9e6a466633eb6ff475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f18dc82d84ba6e698f313b47a849e850(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4713488221168518, 0.20952525734901428, 0.4006534516811371, 0.25041764974594116, 0.20641279220581055, 0.4757500886917114, 0.4089912474155426, 0.37244120240211487, 0.31477871537208557, 0.46037667989730835, 0.3840562403202057, 0.378248006105423, 0.10792902857065201, 0.458946168422699, 0.4390316903591156, 0.32930058240890503, 0.04385751113295555, 0.1507491171360016, 0.48201635479927063], dtype='float32').reshape([19]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_84ece84c9ca4a847b8b015a64f4ec5eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 3136, 3, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_42008e42c0a564bc9128087afff0948a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 3136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 96, 56, 56], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c133d6039a6fd336eedea7099e9c4ce4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 96, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_9887b54065bff26b06e48385f02fa3a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 49, 2, 3, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_23e407b2600c8c5e2eb09a2ac7c0ac73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6524ea711d9138162210764bb5515e99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2776542007923126, 0.19044055044651031, 0.35712218284606934, 0.2734587788581848, 0.32614168524742126, 0.38450565934181213, 0.08328802138566971, 0.44514942169189453, 0.34312331676483154, 0.44081148505210876, 0.4255298376083374, 0.48288875818252563, 0.3824097216129303, 0.4272962808609009, 0.10626713186502457, 0.0953003540635109, 0.2713640034198761, 0.012354888021945953, 0.28227698802948, 0.33888867497444153, 0.038356196135282516, 0.06618242710828781, 0.3468780219554901, 0.041926197707653046, 0.2872498631477356, 0.08407511562108994, 0.34701061248779297], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_85e8663c4b36aacb75e92536f9bdb660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a992bfcdc64b3c4b69c72d45b7f09217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([11, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 320], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_aaa92e05a1f2304ed747468c6ca68a93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 96, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_00cc079be213fb79c00ac3926d465152(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([20, 288, 8, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 288, 192], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_1ad7d345eb8516ff8166b6ce9f27d69d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([20, 288, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 1, 2, 24, 12, 192], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_ff8e3ca1c5cd49b113d97abecbda3aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 24, 2, 12, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 24, 24, 192], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a0203d83cbd15989205660ee4f23d7c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 1, 7, 1, 7, 768], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_00cf3758e42cafe6006e258f6984d9df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 2304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 1, 49, 3, 24, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_86e68d32ef9196aaa5bc0a4b5442afad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.438956081867218, 0.3816528618335724, 0.16485022008419037, 0.34702882170677185], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0a0b1b7e366cbf9e4828ed7aae5a1fda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08836465328931808], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cf41754021c275a0438b57b4b73278e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4312, 16, 2, 4, 6], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_2a23db41565f557bdba7643111b68a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4312, 16, 4, 6], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f311db7c73898b572af69b4603d1ea93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f2cf3ab20302ad1df33bb2cdec87f224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20588138699531555, 0.30630600452423096, 0.26649102568626404, 0.472664475440979, 0.11276429146528244, 0.22665351629257202, 0.04803108796477318, 0.22813323140144348, 0.35285305976867676, 0.4475119709968567, 0.3365159332752228, 0.4718891680240631, 0.47934210300445557, 0.32499727606773376, 0.2692103683948517, 0.2360944151878357], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cce604bbfcf4d8b3a12a0487e8363398(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02642165869474411, 0.08216207474470139, 0.3633526563644409, 0.3227331042289734, 0.31471794843673706, 0.40250590443611145, 0.09814642369747162, 0.4100300073623657, 0.27755844593048096, 0.1596105992794037, 0.3026795983314514, 0.3551560342311859, 0.062424711883068085, 0.05587185546755791, 0.10002518445253372, 0.28775715827941895], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0b1acb06b0664b5c49b3197f81a13f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ca5aa789752502dd1c15f4a41a14d367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb4a6a75d92c2835760d74c78b37e8c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10638931393623352, 0.3899155557155609, 0.3787913918495178, 0.4926273226737976, 0.012389779090881348, 0.3312159776687622, 0.2848368287086487, 0.05988123267889023, 0.3789302706718445, 0.23561517894268036, 0.020840326324105263, 0.1585819572210312, 0.30788975954055786, 0.43448105454444885, 0.30167704820632935, 0.42070573568344116, 0.26632028818130493, 0.14781393110752106], dtype='float32').reshape([18]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7fa020d282a751f5535bb63a5bcd60e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3318941593170166, 0.24705351889133453, 0.3984801769256592, 0.10045222193002701, 0.31767508387565613, 0.09500905126333237, 0.10762251913547516, 0.0847281813621521, 0.41139650344848633, 0.3304615914821625, 0.0874551385641098, 0.11590531468391418, 0.2927607297897339, 0.007297540549188852, 0.017667120322585106, 0.1435425877571106, 0.2661576271057129, 0.38826000690460205, 0.13652725517749786, 0.10768882185220718, 0.3556530773639679, 0.16330434381961823, 0.30451416969299316, 0.4037179946899414], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b603c0fd70d78e6a3f9de386171905ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0c8e14e1da0dd8af4c6bbf543e550ffd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08147464692592621, 0.4115879237651825, 0.4281045198440552, 0.11411646008491516, 0.0697701945900917, 0.40657374262809753, 0.04696368798613548, 0.35068076848983765, 0.09126729518175125, 0.3874135911464691, 0.39716488122940063, 0.10844294726848602, 0.13889899849891663, 0.28850042819976807, 0.4330180287361145, 0.15425673127174377, 0.17423184216022491, 0.40315449237823486], dtype='float32').reshape([18]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_28e8803dfae2bf58a6fec5d379984109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 8, 7, 8, 7, 96], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_f4913bdccc219ee4c1f8ccb9bd7cbd00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 64, 49, 3, 3, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_5e0fbf197bb5e5928d039d339d6b2eeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 7, 7, 768], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3f0388df0dd30b972bcf7d4232770065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17412416636943817, 0.4368537962436676, 0.47844329476356506, 0.2841244637966156, 0.04356177896261215, 0.07672924548387527, 0.0945739895105362, 0.12816616892814636, 0.060536257922649384, 0.3125643730163574, 0.34388467669487, 0.4632628858089447, 0.23395560681819916, 0.02593192085623741, 0.12805302441120148, 0.22401751577854156, 0.2470126748085022, 0.15450827777385712, 0.4084340035915375, 0.08994526416063309, 0.3655916750431061, 0.4603477120399475, 0.10178453475236893, 0.49782589077949524, 0.0718960165977478, 0.36696264147758484, 0.3191574513912201], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d8c3ad013e5e88f8095e74d509cdf926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b9268dd0f88023a42f7d4868e2127cf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 4096], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1280, 32, 128], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09c15212db8e4116d717a2ec81cfd350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8afcac4171042362e2b41d97782c69f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([139392, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3a929403af8a1423214d0afd99e6da1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([34848, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9d7cb5dde6317e9487d24410ecfd6481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([8712, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4dcc60ee618cfe06fff655569c47917f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([2178, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_46272270f5e01a5115d7257d0390c547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([528, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_47c77d8d595097698fbb53426b93d4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 176, 264, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7d8aab9d6f18aad4553aea50b8532d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 88, 132, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_85300b4c002adf9938c9cafa14c44db8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 66, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2932449ca12a9c4437dc3185e7182ea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 22, 33, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e576f72a3569c3b546f79aa95dd88996(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 11, 16, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5301f07d8754be046c939f64ced59ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 176, 264, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_d4eedb08c59264888d4b4cb9b5032739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 88, 132, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_12fdfac20e142eff367441a52bf2c0f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 66, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_290a82df2aa3b1e7729236babaa6626d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 22, 33, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_105899f5edd493c4eb71032a14b38afc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 11, 16, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_15704cb9a3a8f4ff288d07d4310a493a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45081761479377747, 0.13686196506023407, 0.21637752652168274, 0.31419476866722107], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0225f7dc79a0b1409ec0aad788446af7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.459746778011322], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10ee62707b2414e25deb21b51959a133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([576, 96, 2, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 96, 48], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a60266e26e518e4b6a434128a01a16b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([576, 96, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([6, 96, 1, 1, 96, 48], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_5fedf569767caceb27b693b6b9699d9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([6, 96, 96, 48], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_39c6d15321088eec074af2f7e62a775e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1568], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_78e142bc3ad1c821e5123eee45f7f318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1568, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 32, 49, 7, 7], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_aa283396a77a372d81b303b6922ac96c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 25088, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 32, 16, 49, 7, 7], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_141c1f0859b682cf7a65555c133dee49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 512, 7, 7], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c804d74f0649ca974979e2aaba321ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([258], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c804d74f0649ca974979e2aaba321ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([258], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c804d74f0649ca974979e2aaba321ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([258], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_daadad9bf75f5b7f9543f698d1086230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.36395949125289917, 0.3004571497440338, 0.27638670802116394, 0.19535374641418457, 0.004510029219090939, 0.3742370307445526], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_79f66b2ff8d7a5d10932f168dd2485ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c7ce804f9b0f6be3beb30396abf2248
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_b7a6ac4d275337336d1cae685ee2476f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11675350368022919, 0.4269094467163086, 0.12640905380249023, 0.22907717525959015, 0.31840136647224426, 0.21331316232681274, 0.1454918086528778, 0.11361991614103317, 0.2726489305496216, 0.20307783782482147, 0.12221603095531464, 0.4090420603752136, 0.2270182967185974, 0.3963547945022583, 0.06818151473999023, 0.28190624713897705, 0.29520416259765625, 0.04930226877331734], dtype='float32').reshape([18]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d539b7d4220f0f065e383a2bd38d6297(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4191839396953583, 0.12070396542549133, 0.04968828335404396, 0.12514469027519226, 0.08466403931379318, 0.1526838093996048, 0.21586817502975464, 0.49075013399124146, 0.18561631441116333, 0.23484830558300018], dtype='float32').reshape([10]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_032803926fe9e0757089099a73cb58f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3671773374080658, 0.2251962423324585, 0.29521429538726807, 0.36497220396995544], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2745d047ea8ef7c36091383324f4f70b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2138272523880005], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_52262569120aacb495033103ac9436d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 32, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 144, 768], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2ae9927e42eed279b5c723155a005186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 1, 1, 12, 12, 768], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_45c53317adc5f51fca1afaa654702dd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 12, 1, 12, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 12, 12, 768], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d5ed676140fdec2b51d4f6b2f62bb35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([96, 96, 4, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 96, 96], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_741578ae79a5cb82c5e31b597869d922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([96, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 1, 24, 48, 2, 96], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_1739abaeb3c4af56df526a4bec260243(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 48, 24, 2, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 48, 48, 96], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_745002dd0fbd31d31e2449a6dce4fdab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([12, 288, 8, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 288, 192], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3c0c80f30e78e0806a7534c485819e67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([12, 288, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([6, 2, 1, 12, 24, 192], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_c3330ed7e29b339bb9facd519b3d8979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([6, 2, 12, 1, 24, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([6, 24, 24, 192], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5aca3f162b92c881ce73f85cb0fb23d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 8, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a158ef92e0a9b8d42f3f899885bad16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 8, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_3a9546134370028f2020a68712c57988(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 256], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d297b8ecac5dede86a2265e059e20d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49211302399635315, 0.15431459248065948, 0.28994548320770264, 0.2070321887731552, 0.10089040547609329, 0.35877346992492676, 0.03907984867691994, 0.28933998942375183, 0.1337043046951294, 0.3286864757537842, 0.4363308846950531, 0.280532568693161, 0.3004021942615509, 0.21555814146995544, 0.13775525987148285, 0.16165071725845337], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7b677af89d48d4bc610968c9937e6af5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20535996556282043, 0.2290862798690796, 0.3984484076499939, 0.35102611780166626, 0.2331497222185135, 0.45837724208831787, 0.019792502745985985, 0.3102130591869354, 0.34830182790756226, 0.2030649185180664, 0.4910310208797455, 0.019184116274118423, 0.3011069893836975, 0.40667524933815, 0.1559874266386032, 0.36990562081336975], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0b1acb06b0664b5c49b3197f81a13f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2778a9081a66b016e1ac92c85f9465e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 32, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 144, 768], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2d8d95283b6e31b6cbf8fb2e8ce3a8a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([6, 1, 1, 12, 12, 768], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_b1ac169acfa564c8015de7d13cce01da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([6, 1, 12, 1, 12, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([6, 12, 12, 768], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a4fe737c29f2de0610de8192d9e1d805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5bcfc3ca870fd623a05e54a6b74e80c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.040506716817617416, 0.21085339784622192, 0.28114455938339233, 0.3673225939273834, 0.48802638053894043, 0.41977426409721375], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_678eef615c912d53f7af479624724855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ba117c9ce7315fb172057dd48c5e91e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 196, 12, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8676fc8188c783c45ad6d1a6df9cf621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 384, 14, 14], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_76fef51d3105613aa1e0831b91f8fbf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 384, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f2c6be669417fed3e006e1064610b2c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 49, 2, 12, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_81ccaf958fd69fff1a34685f1abdbc54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([960, 96, 2, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 96, 48], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b73572a12b44315c9f7f8a62fc38ef31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([960, 96, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 96, 1, 1, 96, 48], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_418a81dfe181c55753b94749346e9efb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 96, 96, 48], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_65a577376c45138d157a0e25fedcce89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 49], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_65a577376c45138d157a0e25fedcce89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 49], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_65a577376c45138d157a0e25fedcce89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 49], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_65a577376c45138d157a0e25fedcce89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 49], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_65a577376c45138d157a0e25fedcce89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 49], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_65a577376c45138d157a0e25fedcce89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 49], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_65a577376c45138d157a0e25fedcce89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 49], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_65a577376c45138d157a0e25fedcce89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 49], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a89b4955c6f624dd909c682bbf9f8262(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47828707098960876, 0.23036950826644897, 0.04434008151292801, 0.17016369104385376, 0.16162338852882385, 0.2887677550315857, 0.15269707143306732, 0.2701972723007202, 0.080161452293396, 0.00814066082239151, 0.28985682129859924, 0.4188685417175293, 0.13493658602237701, 0.4167904257774353, 0.3330402374267578, 0.16454735398292542, 0.3396787941455841, 0.21985498070716858, 0.003296131733804941, 0.3775654733181, 0.4967746436595917, 0.4952768385410309, 0.054770126938819885, 0.3809250593185425, 0.2339312583208084, 0.4092939794063568, 0.028206296265125275, 0.28066307306289673], dtype='float32').reshape([28]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ff91a87f51b71fdd01632089828fcb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 3, 4, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_74aba368c58a586a959f54a95125bbe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 4, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 128], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_0096d238dbaa46245b76f8d19571151f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0096d238dbaa46245b76f8d19571151f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0096d238dbaa46245b76f8d19571151f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0096d238dbaa46245b76f8d19571151f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0096d238dbaa46245b76f8d19571151f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0096d238dbaa46245b76f8d19571151f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0096d238dbaa46245b76f8d19571151f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0096d238dbaa46245b76f8d19571151f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5564a0136ec24b07b51c5746026833ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3801538348197937, 0.45731520652770996, 0.14592859148979187, 0.19617250561714172, 0.12918847799301147, 0.4743257164955139, 0.39642760157585144, 0.36924973130226135, 0.3928917646408081, 0.3571273386478424, 0.26951515674591064, 0.2456790953874588, 0.058116670697927475, 0.429230272769928, 0.3165701925754547, 0.39761316776275635], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7f66492ec8fb30185c7f352fd4519a7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 38, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8f4b87f8cfe02680f4925ec39785ca39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2383de463e8f2cb1c12d3021a2bbaf09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 38, 84], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 21], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e9f5eaad6857b7672a4fe04e448c79f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28163161873817444, 0.3863672912120819, 0.40511223673820496, 0.15598897635936737, 0.40265604853630066, 0.2268117070198059, 0.06514408439397812, 0.13565950095653534, 0.44213563203811646, 0.36502283811569214, 0.3574792146682739, 0.2477431744337082, 0.09224268794059753, 0.2151021957397461, 0.4989434778690338, 0.45762795209884644, 0.47346025705337524, 0.20975224673748016, 0.12760363519191742, 0.36555156111717224, 0.11410477757453918, 0.14366517961025238, 0.20343025028705597, 0.07436370849609375], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7f5f01061f2fcfb8daa728cfe7b13a71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 19, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_c24f1b8da44063b73d61072f9d1cbc02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([126], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b3ce5ac91ac4f895334a60d60d1b8469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 19, 126], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 21], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_1f6251d289db166d2fafa1f5e5ffaa73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2729511857032776, 0.4803543984889984, 0.48965469002723694, 0.22982783615589142, 0.06872314214706421, 0.06946268677711487, 0.17774717509746552, 0.17530390620231628, 0.4079574644565582, 0.1728995442390442, 0.23499657213687897, 0.4048655033111572, 0.46517932415008545, 0.3223994970321655, 0.37012726068496704, 0.12389199435710907, 0.49085065722465515, 0.25943559408187866, 0.18701542913913727, 0.24375201761722565, 0.07687385380268097, 0.4178529679775238, 0.057761114090681076, 0.10172946751117706], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2515e87374c13db108081e4d7242fcb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 10, 10, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_c24f1b8da44063b73d61072f9d1cbc02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([126], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b1c31f25cad863a73d3afaa1ce99f1d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 10, 10, 126], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 21], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3057be94770c2d710e5599f8aeddcf2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14768868684768677, 0.22790661454200745, 0.4400694966316223, 0.324767142534256, 0.23799774050712585, 0.1804571896791458, 0.04167087376117706, 0.01567375473678112, 0.2369304597377777, 0.2242467850446701, 0.35853901505470276, 0.32035040855407715, 0.11489180475473404, 0.49122723937034607, 0.26480069756507874, 0.03675248101353645, 0.0781988874077797, 0.4675684869289398, 0.3752947151660919, 0.0735933855175972, 0.2293705940246582, 0.19286184012889862, 0.3621313273906708, 0.42052745819091797], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9c89e5f1c056eb4a4638ca89fc878a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 5, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_c24f1b8da44063b73d61072f9d1cbc02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([126], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e715ab314bd2188edcec026ec3ccec26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 5, 126], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 21], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_d80e93acf61ee41e7b615dd4a17df76d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.030849209055304527, 0.21966761350631714, 0.3220648169517517, 0.03864293918013573, 0.35507577657699585, 0.15690089762210846, 0.07714590430259705, 0.1670490950345993, 0.05313710495829582, 0.36865636706352234, 0.0382884182035923, 0.08356686681509018, 0.38652679324150085, 0.1700289398431778, 0.3254833519458771, 0.4098016619682312], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f24c90709fb51890abfc04f08bb6d7e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 3, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8f4b87f8cfe02680f4925ec39785ca39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bed1ac4527fd8d6b04381baa67bbc6a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 3, 84], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 21], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2b2be481727dbf60f14371725ff4e0b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01352987252175808, 0.19377119839191437, 0.31270864605903625, 0.06185305863618851, 0.300590842962265, 0.11447631567716599, 0.09409885108470917, 0.3769531846046448, 0.3099818229675293, 0.05466809123754501, 0.34677764773368835, 0.46176815032958984, 0.1946910172700882, 0.4882251024246216, 0.39294615387916565, 0.3196198046207428], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_99989f06cd5c8128506e54f04a92b2cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[16.16293716430664, 15.970179557800293, 16.368606567382812, 14.714933395385742, 17.15117073059082, 16.008346557617188, 16.108627319335938, 15.955602645874023, 16.818113327026367, 14.214376449584961, 16.382869720458984, 16.872631072998047, 16.596763610839844, 16.210844039916992, 16.85279655456543, 16.025291442871094]]]], dtype='float32').reshape([1, 1, 1, 16]),
            paddle.to_tensor([0, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8f4b87f8cfe02680f4925ec39785ca39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1fd2d5aa20667873c2ce80c44664507b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1, 84], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 21], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f03781eaa67009c7e9c7a9d3e0791d73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([6888], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f03781eaa67009c7e9c7a9d3e0791d73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([6888], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_10836e9344f5fbea86733d16ccd825de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([130872], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4, 19], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f03781eaa67009c7e9c7a9d3e0791d73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([6888], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_383ac114a22d94a439784662ffdd3c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.46426013112068176, 0.1539950668811798, 0.29448407888412476, 0.15874990820884705, 0.1506706178188324, 0.38457056879997253, 0.27485886216163635, 0.3633710741996765, 0.1337706744670868, 0.29570865631103516, 0.13407304883003235, 0.19589920341968536, 0.26694056391716003, 0.24831783771514893, 0.11837150156497955, 0.3068545162677765, 0.04680236428976059, 0.3562089502811432, 0.4646495282649994, 0.23689784109592438], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3aa6732f3220b71947e43a708ca66870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35860955715179443, 0.1383810043334961, 0.2875531017780304, 0.4655022919178009, 0.4516802430152893, 0.3055037260055542, 0.0621982105076313, 0.36679255962371826, 0.09523655474185944, 0.351299524307251, 0.1326848417520523, 0.4894323945045471, 0.37123343348503113, 0.25027498602867126, 0.11418075859546661, 0.3026505410671234, 0.38298436999320984, 0.4103942811489105, 0.018619298934936523], dtype='float32').reshape([19]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2759dbea1768c6b8b61c1a0f99173554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2886560261249542, 0.25516805052757263, 0.37995588779449463, 0.1866346150636673, 0.011833615601062775, 0.44703856110572815, 0.38019421696662903, 0.2674083113670349, 0.16704218089580536, 0.48892858624458313, 0.35689467191696167, 0.3630383610725403, 0.05709678307175636, 0.4513890743255615, 0.423170268535614, 0.08083056658506393, 0.4084092080593109, 0.22413676977157593, 0.44084152579307556], dtype='float32').reshape([19]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8ab4441f85b1f3ea587b0742235905dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([2112, 96, 2, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 96, 48], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_603a86b03c6f2bcee4184495ba49336c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([2112, 96, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 96, 1, 1, 96, 48], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_255cc3be62c9ed007882d5337d63f88f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 96, 96, 48], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_026b45f05ae910fdc11eba3224cc4fa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 28, 50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 2, 36, 28, 50], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_f3f757432e02e796df00e73272ec1bd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 2, 28, 50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 72, 28, 50], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_353b5490a9358d66c51c1193073748c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47732844948768616, 0.01131041906774044, 0.1485288292169571, 0.14607371389865875, 0.24204109609127045, 0.05766914784908295, 0.35664084553718567, 0.3605709969997406, 0.03912150859832764, 0.08118315041065216, 0.4907556176185608, 0.32687848806381226, 0.03591075912117958, 0.4724140167236328, 0.4766179025173187, 0.4473150074481964, 0.03255652263760567, 0.3339284360408783, 0.2628086507320404, 0.26661086082458496, 0.31822651624679565, 0.3102588355541229, 0.06835657358169556, 0.15071336925029755, 0.069854736328125, 0.2278912365436554, 0.32346007227897644], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f7f62f2f0f020f9e6a466633eb6ff475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a9cce944cc27f90184534ddd3a4663e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3344857394695282, 0.31709152460098267, 0.2157890945672989, 0.2358112633228302, 0.4190995395183563, 0.18012894690036774, 0.0843162089586258, 0.375400573015213, 0.15374331176280975, 0.17527948319911957, 0.10777634382247925, 0.2217588722705841, 0.021053466945886612, 0.40669897198677063, 0.1216207817196846, 0.09440657496452332, 0.10163816064596176, 0.3015816807746887, 0.11409579962491989, 0.1265624761581421, 0.2281494140625, 0.1909673511981964, 0.05752779170870781, 0.018506621941924095, 0.38144227862358093, 0.145588681101799, 0.47904303669929504], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d8c3ad013e5e88f8095e74d509cdf926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_768d3a7c9b696ce8e4751c1d76c0c365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 3200, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 32, 100, 2], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3347fd542b487a4dd18382464c9e9e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13873741030693054, 0.3854258060455322, 0.18275712430477142, 0.43969258666038513], [0.13893212378025055, 0.18957515060901642, 0.38497790694236755, 0.3787880539894104], [0.2273373305797577, 0.17694276571273804, 0.13323169946670532, 0.4903319180011749], [0.17077070474624634, 0.0017483800183981657, 0.19547641277313232, 0.11881936341524124], [0.2892541289329529, 0.26534363627433777, 0.37189048528671265, 0.18562017381191254], [0.25426748394966125, 0.07485723495483398, 0.42880791425704956, 0.12537863850593567]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([0, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cd664a662fbbea631f3d56c19bbd54c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2beca1fd743440df3e6f0bcff7386dd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4116, 4, 17], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_39821081d5e8beddb5b209c4c2412080(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.41461750864982605, 0.3710567057132721, 0.05767848342657089, 0.488178551197052], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a033d50c133866f680164ba3eec6330a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43095144629478455], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_fc600c8fa3f915d019f1bbc6010615c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30721530318260193, 0.2988300621509552, 0.0658896416425705, 0.0999918282032013, 0.09717690199613571, 0.2179814577102661, 0.041885267943143845, 0.4674232602119446], dtype='float32').reshape([8]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e49831eead588f3782b29e82264473e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28353458642959595, 0.004079361446201801, 0.4338584840297699, 0.24296773970127106, 0.05032340809702873, 0.4947565197944641, 0.49901580810546875, 0.2739507257938385, 0.2655699849128723, 0.2828238606452942, 0.4866964519023895, 0.496524840593338, 0.26521965861320496, 0.27649638056755066, 0.008649447001516819, 0.46980875730514526, 0.28374212980270386], dtype='float32').reshape([17]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8dccc8bc5773bc4065087b4e7bb99306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_057e8959f882fd1f5c4016eeb76c0ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_177e893c66b4d36e329fd8626eee508b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0ad80d916458308fcc83eb19dfb9044d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_057e8959f882fd1f5c4016eeb76c0ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_177e893c66b4d36e329fd8626eee508b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0ad80d916458308fcc83eb19dfb9044d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8333c7ed7866a2e3ce0fc4df296f0fd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 49, 8, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_962579d682c78d2b29811b675ca6cc39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([8, 2401], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 49, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10569d43da2534096949b28c39ebd1af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0199394803494215, 0.019288457930088043, 0.032455794513225555, 0.4119749367237091, 0.3542870879173279, 0.00468840217217803, 0.08049707114696503, 0.15488936007022858, 0.1621120721101761, 0.08285702764987946, 0.16398707032203674, 0.24245622754096985, 0.2437116950750351, 0.38498586416244507, 0.3577861189842224, 0.15404781699180603, 0.48698529601097107, 0.06458936631679535, 0.1483536958694458], dtype='float32').reshape([19]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_232589db835d21d5d459f2c6dc67aad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 512, 1024], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6b48af10cfc597a700390a10fcce337f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39974749088287354, 0.33195382356643677, 0.4932257831096649, 0.36529600620269775, 0.1280062347650528, 0.43135353922843933, 0.10572940856218338, 0.3815785050392151, 0.473796010017395, 0.13703569769859314, 0.2906275987625122, 0.02393452450633049, 0.32638946175575256, 0.38457661867141724, 0.16773973405361176, 0.35485613346099854, 0.070821113884449, 0.35504794120788574, 0.4888022243976593, 0.18093229830265045], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c20bfa7ef2d90682042227d391eb0b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 196], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c20bfa7ef2d90682042227d391eb0b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 196], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c20bfa7ef2d90682042227d391eb0b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 196], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c20bfa7ef2d90682042227d391eb0b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 196], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c20bfa7ef2d90682042227d391eb0b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 196], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c20bfa7ef2d90682042227d391eb0b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 196], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c20bfa7ef2d90682042227d391eb0b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 196], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c20bfa7ef2d90682042227d391eb0b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 196], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1643e3b2c4671f1b072c81916e399e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1643e3b2c4671f1b072c81916e399e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c963c91c70ab9afc497667cdb11e99e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.441054105758667, 0.38664939999580383, 0.28109169006347656, 0.08136561512947083, 0.3329803943634033, 0.3670419156551361, 0.494271844625473, 0.06330656260251999, 0.4279530644416809, 0.11804855614900589, 0.04624996706843376, 0.43904972076416016, 0.23624201118946075, 0.2734617590904236, 0.47509485483169556, 0.028826354071497917, 0.12327395379543304, 0.37450334429740906, 0.05560190975666046, 0.04735464230179787, 0.3233475983142853, 0.416772723197937, 0.3536181151866913, 0.40622520446777344, 0.4398817718029022, 0.4188087284564972, 0.27352389693260193], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f7f62f2f0f020f9e6a466633eb6ff475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_945fca59373e41f913991129f8a1c012(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 56, 56, 96], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f311db7c73898b572af69b4603d1ea93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2ef9510ff6456df1e1bfa724e158f8ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4272748529911041, 0.29769858717918396, 0.15885521471500397, 0.04473520815372467], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e641b8c746ed91af9c97d8b49f9f2bfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3840104043483734], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2a4a85dc6f2ea87cc62ddbafa339c506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4502294361591339, 0.27858400344848633, 0.43629756569862366, 0.24267302453517914, 0.47881069779396057, 0.42316505312919617, 0.1957198679447174, 0.19584821164608002, 0.13948386907577515, 0.26827800273895264, 0.21529464423656464, 0.25985562801361084, 0.34080344438552856, 0.15619662404060364, 0.12815794348716736, 0.11673469841480255, 0.2506316006183624, 0.17910820245742798, 0.3811485767364502, 0.14674115180969238, 0.09395298361778259, 0.10557638108730316, 0.18883784115314484, 0.08551861345767975, 0.41333791613578796, 0.43984031677246094, 0.05554024875164032], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f7f62f2f0f020f9e6a466633eb6ff475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_faab50eb1634d23a21830a8e3c0ddf64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([392], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_315616a79e1cc4c2d6e0a0defb0717ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 392, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 8, 49, 28, 28], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_bcf4e8da9dcb4d46bfd885a1555d3834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 6272, 784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 8, 16, 49, 28, 28], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_4e6f52a10167ea3ad440e5912e662ad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 128, 28, 28], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9fc4c0ae18a1f17b0a2d54a9fb04cefb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 205923, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43e7f58ac9c792e85f68a7d2fac9676e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 205923, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_582c4d804520c04931648161479d67ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 192, 28, 28], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f561dfce2dc5f61884817f6dedcfa452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 384, 14, 14], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4ee3b807c30c71482b85c15453249e3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f10b91eeaa45ba5106bd12c92acf2d26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_940b65524433657086eff3256fa78c3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_47f86e9ad338f10a9b7bfe277726fbb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 6069, 4, 17], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f8a8f559b0a14017fc840c5e509c3a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([6260], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f8a8f559b0a14017fc840c5e509c3a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([6260], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_55e9f4742a573d9e2e7d1bfda5cd9a5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([106420], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4, 17], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f8a8f559b0a14017fc840c5e509c3a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([6260], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5630546e786d9542b4aa8aec28e48946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5648fb57a7fc0f60bdde9ea58affba12
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3549], dtype='int32'),
            paddle.to_tensor([1, 3549], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_99f23fc5b1c08f7edaaf85c8cfff1254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.11793242394924164, 0.4516194462776184, 0.376777708530426, 0.3160300552845001], [0.14109523594379425, 0.44240888953208923, 0.15395429730415344, 0.3476671576499939]]], dtype='float32').reshape([1, 2, 4]),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6bc8644aff1f01cfae9aeb2ad508591c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3549, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_02cfadd927e4ab82bf663ebb392d603b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3d93e12b3a6b7bc54baaab35038c4945(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 784, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 16, 49, 14, 14], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_7edd52de12b09b8d1b7cb89f835f5858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 12544, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 16, 16, 49, 14, 14], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_212e682f16e7d483ae65a8efcb4cac8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 256, 14, 14], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b603c0fd70d78e6a3f9de386171905ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1d35570dadea980ad61998cdd7c6f5d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8396af63fa7f79de582f076e00a06985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 28, 28, 192], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9f661a6662eb24bb336d478fed80bb27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 512, 4, 8, 16, 8], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_f3c3d9f1bc18b300d9ab85cf2268ec92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 8, 512, 4, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 512, 4, 16], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_24f6e6b0f28b11f77f090ffd3648a52a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3388068675994873, 0.48748448491096497, 0.14841881394386292, 0.30166947841644287], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_fb9d8b660f658ac70a92db2e764d27b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25633734464645386], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0755dfd9a6c93b7f2ea09b68a399eb9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 4, 7, 4, 7, 192], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_58010b0bce08203f99bde903e68d7c27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 16, 49, 3, 6, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_e6c57ecc1f1cd3c57359c0f0c28ac60a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17646734416484833, 0.15945705771446228, 0.08413650840520859, 0.20054583251476288, 0.48495227098464966, 0.4162448048591614, 0.20483878254890442, 0.31059718132019043, 0.03976629674434662, 0.1022886112332344, 0.18222478032112122, 0.3260064125061035, 0.3508162498474121, 0.021887443959712982, 0.08588871359825134, 0.4037362337112427, 0.4869781732559204, 0.41016685962677], dtype='float32').reshape([18]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ca5aa789752502dd1c15f4a41a14d367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_57729a834dd7c0862c604198be57a021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 1, 7, 7], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4b0c8e32e7cd640381c3c418301a3e2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 4, -1, 13, 19], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_c45b5a5c7b65b0202f09152e8c4dc774(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 20, 13, 19], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ddee2c3bb8e43e3c4c2e063571c3cac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15311741828918457], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_556ee7e70131469d55a627afc112b9cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 104, 101], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 8, 52, 8, 202, 8], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_c519673e55250adf2dbc5bf16ea53eb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 8, 8, 52, 202], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 512, 52, 202], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c0156cfed6741fd3941c393340f4d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f80261241dca370c98381bc30b86e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_fdfda1ca10146024a30c43f6096d45cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([182400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3ad8b202bf54e78506ea3ea2dd90b345(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([45600, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fa60ce92beddca5aba33f12823929975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([11400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1c1108d0ef6567b7ff2b4851f80b26a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([2850, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3356a8e8f7f2192663dfcff216ddeea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([741, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7f6909467fa190287930e51096fedd3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 304, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_79c4d415751b16b36c77ec45811dadec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 152, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_86e1111ed9e782e3e93da9cd0e81c88e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 76, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_fa431db6b95bd147bb889903b4363897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 38, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5687918f27047ffb9211b2b21eddb9d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 13, 19, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_391453452dbd65a4e519bb9004a5cbfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 304, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_27fe08e6f2a89252a3d24fb64caf7e95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 152, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e9f7cd8c278ae3f7ebac8bfbd047ae5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 76, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_608b70d5cd42285b49591fccd69dfcf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 38, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_0e83cfaa07a3e88f153d64ba1c4cbcdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 13, 19, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_75cd851225de23b977a5c8a782af3b13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4670657813549042, 0.0076882424764335155, 0.03990844637155533, 0.2652323544025421, 0.09579344838857651, 0.08974035829305649, 0.23166602849960327, 0.35530877113342285, 0.44746220111846924, 0.48748281598091125, 0.08330211788415909, 0.1884688436985016, 0.14912939071655273, 0.20113375782966614, 0.11553351581096649, 0.1984786093235016], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a4fe737c29f2de0610de8192d9e1d805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8b2634f6bfb16a06e2f623a4a12b7dd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3548639714717865, 0.14659830927848816, 0.4033186435699463, 0.2505156099796295, 0.40094032883644104, 0.07909703254699707, 0.1168547123670578, 0.1578483283519745, 0.2989654242992401, 0.44123607873916626, 0.07852701097726822, 0.2481130063533783, 0.04536408931016922, 0.06156700849533081, 0.14479082822799683, 0.408723384141922, 0.27076399326324463, 0.002323882421478629, 0.22698870301246643], dtype='float32').reshape([19]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9534c04a1fcba36543a762032e49ef67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1640213578939438, 0.23343661427497864, 0.34252065420150757, 0.2556758522987366, 0.371285080909729, 0.2500620484352112, 0.28158268332481384, 0.23829731345176697, 0.036219630390405655, 0.14533396065235138, 0.09062109142541885, 0.3174276351928711, 0.41852572560310364, 0.4068596363067627, 0.2567237913608551, 0.27359452843666077, 0.4012127220630646, 0.06986651569604874, 0.3333585858345032], dtype='float32').reshape([19]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb987f2d57c442af87040e45b7bf5196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3022405803203583, 0.13975271582603455, 0.2562035918235779, 0.165739506483078, 0.4747582972049713, 0.2216244339942932, 0.32729798555374146, 0.4953100085258484, 0.19836916029453278, 0.23722738027572632, 0.41624271869659424, 0.32826900482177734, 0.03817278519272804, 0.2691551148891449, 0.3417932391166687, 0.4307059049606323, 0.42065170407295227, 0.08552996814250946, 0.37214604020118713], dtype='float32').reshape([19]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8fd54cbd3bfe85dc57b3c3f0a4a17107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34713220596313477, 0.04512999951839447, 0.1399756371974945, 0.3063362240791321, 0.39673566818237305, 0.27475467324256897, 0.49128106236457825, 0.45526859164237976, 0.0769863948225975, 0.2389577329158783, 0.2743740379810333, 0.15987145900726318, 0.07612191140651703, 0.48460090160369873, 0.46621766686439514, 0.028033284470438957, 0.022582566365599632, 0.08579214662313461, 0.06570130586624146], dtype='float32').reshape([19]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d1bd988bacc9097b839c0761d5187949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 1025, 3, 6, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_1b2117faef3481758d1185ab89964bf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 6, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 1025, 384], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_0afccbd4bcaa90ad00c9ceacaad8c390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35281938314437866, 0.06439217180013657, 0.09660796821117401, 0.1536925882101059, 0.4763009250164032, 0.41743549704551697, 0.4866028130054474, 0.11119130998849869, 0.42410922050476074, 0.4218108057975769, 0.3781158924102783, 0.1703604757785797, 0.35809600353240967, 0.20729130506515503, 0.440778911113739, 0.29840797185897827, 0.22010323405265808, 0.1677442491054535, 0.41721779108047485, 0.1553977131843567, 0.2600405216217041, 0.4292179346084595, 0.14561370015144348, 0.1780981868505478, 0.2546343207359314, 0.290523499250412, 0.39489370584487915, 0.10626474767923355], dtype='float32').reshape([28]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d6e50ec5744fef11d06f34b9ba3d3039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 123783, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f006fc1be5b1bda42fdccd0c91874b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 123783, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_39618cde2d3fe0e68d04413d5d77c7d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 64, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_39618cde2d3fe0e68d04413d5d77c7d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 64, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b235860d14a565ae3551e71be4480c65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 512, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7df3ccc1ddeb8b527b58783fcb59faeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 512, 64, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9eba28fe9cf2cd455c7b6b0f8da59325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43477874994277954, 0.3511645495891571, 0.2642982602119446, 0.05791514366865158, 0.32713043689727783, 0.26966115832328796, 0.24919743835926056, 0.13156291842460632, 0.4441141188144684, 0.37118133902549744, 0.23394842445850372, 0.30111125111579895, 0.23570385575294495, 0.2626310884952545, 0.46870142221450806, 0.2593550682067871, 0.2086813747882843, 0.18825149536132812, 0.4025612771511078, 0.21322013437747955, 0.2170015573501587, 0.382428914308548, 0.4515968859195709, 0.431020587682724, 0.3365554213523865, 0.4770755171775818, 0.11425936222076416], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d8c3ad013e5e88f8095e74d509cdf926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_493267d44bb519334849121b432934f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 640], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 196, 8, -1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_02cfadd927e4ab82bf663ebb392d603b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3d93e12b3a6b7bc54baaab35038c4945(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 784, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 16, 49, 14, 14], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_7edd52de12b09b8d1b7cb89f835f5858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 12544, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 16, 16, 49, 14, 14], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_212e682f16e7d483ae65a8efcb4cac8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 256, 14, 14], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6082e9fe22f8056e00ce78d30aebbb28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 1, 5, 5], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d9893b0a55c383a51f55e8cbf427910a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21791139245033264, 0.1414179801940918, 0.3480929434299469, 0.2139100581407547, 0.3840459883213043, 0.18153390288352966, 0.35136187076568604, 0.09536262601613998, 0.19064314663410187, 0.4711149334907532, 0.2386804223060608, 0.14812904596328735, 0.1314697265625, 0.40142548084259033, 0.11835330724716187, 0.20373603701591492, 0.3831857442855835, 0.16392728686332703], dtype='float32').reshape([18]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6e82786a8ccd1ece16eba66b2ea2644b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08228106051683426, 0.4774888753890991, 0.34013333916664124, 0.1820846050977707, 0.35879412293434143, 0.008671501651406288], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0daca6953cd44ba77d5ea4d98e11e46a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.46089258790016174, 0.3948666751384735, 0.3953385055065155, 0.4432721436023712], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9aed1a6682f5bc21f1080b9f5254faeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39090412855148315, 0.4886815547943115, 0.465143084526062, 0.01248091273009777, 0.4762343466281891, 0.03402049094438553, 0.20702587068080902, 0.48705440759658813, 0.4302976727485657, 0.0919407308101654, 0.29588794708251953, 0.3755166232585907, 0.12420680373907089, 0.016455508768558502, 0.10923026502132416, 0.010181121528148651], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_709acf07bbf85c2100c04377e92bffe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 171888, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_29d30d263ee52de04dc52600013f7b3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 171888, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5442f0f2b647eeab7f0512395152951b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 49, 24, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_28e495d1af865cff25f7e13cfba0df28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1536], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 49, 2, 24, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_5da7b4cb2bce4ffb80abfdd4c841764d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 16, 64, 150], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_911af01aeab50042d07d2017e0b74f38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([14, 14, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bb18ba4dc433acd7c623f115c89f531e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([14, 14, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4a7dbd32fde1aa6346b09d0d99b8c5cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([28, 28, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_12f31f264665fa85e1a466eca083aed9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([28, 28, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5e1f72353c97b60e31c3b712a27d3591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([56, 56, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d2abfe37a04eeb14256ee1cf5fb491da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([56, 56, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_47d382c8725f174f4e0a45a35ef04f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07331360876560211, 0.21548666059970856, 0.2241717129945755, 0.26337406039237976, 0.07473746687173843, 0.1414179503917694, 0.23187090456485748, 0.4820196032524109, 0.21851655840873718, 0.23503901064395905, 0.07695723325014114, 0.4980725944042206, 0.2969333529472351, 0.3479432463645935, 0.25253182649612427, 0.08917839080095291], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a9831565bc626516c4870f0f1a47f015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39044249057769775, 0.30304643511772156, 0.022728687152266502, 0.40620189905166626, 0.06435657292604446, 0.2597275972366333, 0.02079247310757637, 0.48708513379096985, 0.2609095275402069, 0.022945355623960495, 0.4933762848377228, 0.3163323700428009, 0.47774550318717957, 0.052629757672548294, 0.3717487156391144, 0.15716631710529327], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0b1acb06b0664b5c49b3197f81a13f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ca5aa789752502dd1c15f4a41a14d367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0edc0a743c5d6db5f21baf1ffe5aeb9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([65280, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a9c928d83bfcd8485e44bc9b0912dba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([16320, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_715b6faecded53a35c3ebb1c52bc8b7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([4080, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e80115e10cc4997b1309241cb91a10d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([1020, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_346259e1d9f1bde9d2c7100f6d36f8e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([270, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a176d43ebdcbdfb88293b87dc64f6f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 136, 160, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_741ba512a0b2aaac71adfc23602de329(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 80, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f44855db0d9dea9f2dc7a1a607f80958(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 34, 40, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2000f803e5388f76cda0b20b1d1d4ae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 20, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3737b5248301818e686fe90d53061526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 10, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_ce1a352a2cf319740a84474daffc292c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 136, 160, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_918dc958d17d25f17edccabc40d94026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 80, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_695c75a02722d4d4aa5909de51a7fc99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 34, 40, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_57fca17d80d8e812d6db8e0ca51a9705(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 20, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_95724c80a803dae399eb7025858acff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 10, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c7e55b7e99e347e28038422f8c04d602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([64, 512, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 4, 16, 512, 8, 8], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_382f1afe945e8b7edca2446396a1da67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 8, 16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 512, 32, 128], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_918e13c281a13bd80608d45c9e0ac990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 14, 14, 384], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f154fb6a5f708a24c311742980220d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 128, 4, 80], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a4fe737c29f2de0610de8192d9e1d805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ada00726f6891244426d06e940f73144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([169, 512, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 13, 13, 512, 8, 8], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_0896d6c63fade97951675ad627cca62d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 13, 8, 13, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 512, 104, 104], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_42008e42c0a564bc9128087afff0948a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 3136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 96, 56, 56], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0f4352960c65a16f99e58f897e0fea5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5648fb57a7fc0f60bdde9ea58affba12
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4116], dtype='int32'),
            paddle.to_tensor([1, 4116], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d2d02d7889eeff4f6c2bea377e71fb5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.28155791759490967, 0.13601340353488922, 0.3007742166519165, 0.04468493163585663]]], dtype='float32').reshape([1, 1, 4]),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9baf0f76f67bbffc714ed8f4e6f7b3eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([4116, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 4116, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_67885294bc04c397eaa071255384cd0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5bc320b26e1dd96923a1d619e75f258f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([400], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_29222129b1825b5f4b10747ab3b3e53e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2048, 5, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45f1368c8dc8d2db88c42226cd7eb179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 2048], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 160, 32, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f80261241dca370c98381bc30b86e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_139ed0fed2774f5a2d8d1ce8bd2de244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 160, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_dd6406f187a484419b57961cba250b32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 5, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_ef07987a13c22e9a2cb83ce2f77a6ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 5, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2048, 160], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_88d54681935c0c8fa28bc98fda9b3dc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 217413, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_489fe25119dcf5b8184ae89a0e2bf85c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 217413, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c941baf3f3eb9e09cf4565b4dbe30212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1024, 8, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cd9f7710b8e6ca507715e11de65fc1f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 8, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_7f6cab845257997e4d6b2ad3200cb8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1024, 256], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e40d2a2ba530a7bf1272627c7c089840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.139553964138031, 0.346799373626709, 0.0536353625357151, 0.2850117087364197, 0.27531304955482483, 0.10546548664569855, 0.08438800275325775, 0.2899167239665985, 0.007388056721538305, 0.11100167781114578], dtype='float32').reshape([10]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_144e103f07d6ad1fc3baf1cbd84c0e52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27082231640815735, 0.33490046858787537, 0.2129560261964798, 0.44473811984062195, 0.4280931353569031, 0.40926671028137207, 0.08794326335191727, 0.30876418948173523, 0.4898565113544464, 0.47698524594306946, 0.22748956084251404, 0.14915533363819122, 0.38840267062187195, 0.05443954095244408, 0.2606116533279419, 0.38114622235298157, 0.20743988454341888, 0.3013095259666443, 0.445746511220932, 0.24819695949554443, 0.4909219741821289, 0.17167975008487701, 0.19772426784038544, 0.2709018588066101, 0.005016819573938847, 0.47383913397789, 0.35156330466270447], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f7f62f2f0f020f9e6a466633eb6ff475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ca5aa789752502dd1c15f4a41a14d367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_425808648cd2246dbc7a86ccfb8db8bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16, 32, 64, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7b8f2f063dd80c6a7b139facb8b8692c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 64, 8, 25], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_71f150460cf4aeca6f1797e61a91a15f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 784, 6, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ab89296a48027a3c8439ae8d28d21342(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 192, 28, 28], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_dcb22da66a9b4ec5a98fdcbe05f769eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 192, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a818699fc07f4cb5234f00baa24247dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 49, 2, 6, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_e5ec1e605f87a274c28649cf9bc487d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f311db7c73898b572af69b4603d1ea93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_327faf802b29c908736722b81481d380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4090927541255951, 0.19149042665958405, 0.33582037687301636, 0.32508769631385803, 0.007077286019921303, 0.2533668577671051, 0.34066158533096313, 0.1743483543395996, 0.14922216534614563, 0.22143197059631348, 0.13888531923294067, 0.41043978929519653, 0.4028628170490265, 0.0771675780415535, 0.11641807854175568, 0.29543858766555786, 0.04362637922167778, 0.15817692875862122, 0.2164444625377655, 0.1782871037721634], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8e795fa47835ae789ee5ee4d209754a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([43, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 320], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8f4b87f8cfe02680f4925ec39785ca39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d07db4fbfad8a4199c3c232094ee1080(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_40f112bbf4fdfa99815543a3407fe62c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_99ae158cd42fa4ad720e32814e64d535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33411356806755066, 0.20749329030513763, 0.4857958257198334, 0.4400416314601898, 0.031147269532084465, 0.3495064675807953, 0.18948426842689514, 0.019732754677534103, 0.19905009865760803, 0.2749037444591522, 0.27897581458091736, 0.4502950608730316, 0.2804824709892273, 0.48003101348876953, 0.2864794135093689, 0.49860823154449463, 0.238205686211586, 0.49859797954559326, 0.014626484364271164, 0.23257531225681305, 0.3710925281047821, 0.2264409363269806, 0.4479336142539978, 0.33939018845558167], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f94bff6c4517cfdde825cbbec4f89936(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.059485211968421936, 0.47505685687065125, 0.1970706433057785, 0.36402958631515503, 0.38056623935699463, 0.34431007504463196, 0.29810234904289246, 0.36643823981285095, 0.1777384877204895, 0.34239932894706726, 0.3328477442264557, 0.43187013268470764, 0.19761282205581665, 0.28306126594543457, 0.24166440963745117, 0.04327418655157089, 0.17782773077487946, 0.43242380023002625, 0.17548276484012604, 0.3483864665031433, 0.31130707263946533, 0.20974984765052795, 0.46635565161705017, 0.08552739024162292], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5cf22b11217a57e0c42968c141b24197(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15950272977352142, 0.05581577867269516, 0.330342173576355, 0.24772517383098602, 0.05522269755601883, 0.4921290874481201, 0.1203453540802002, 0.33430102467536926, 0.1776944100856781, 0.29580157995224, 0.35849013924598694, 0.16003961861133575, 0.38340669870376587, 0.31682708859443665, 0.3825153112411499, 0.24645648896694183, 0.05583714693784714, 0.3470948040485382, 0.37657999992370605, 0.19377607107162476, 0.38621601462364197, 0.20142687857151031, 0.48075079917907715, 0.0326036661863327], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d6b89d38adafc69996cd6b5b9967b66b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 464, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 232, 16, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_b020f64f00db9dc215a7598f858dddd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 232, 2, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 464, 16, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9580a70109e988ed5b085d21dd89de63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3888380527496338, 0.28277847170829773, 0.21224597096443176, 0.2662551701068878, 0.008501093834638596, 0.4443865120410919, 0.03771897777915001, 0.10867200046777725], dtype='float32').reshape([8]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_faab50eb1634d23a21830a8e3c0ddf64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([392], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_315616a79e1cc4c2d6e0a0defb0717ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 392, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 8, 49, 28, 28], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_bcf4e8da9dcb4d46bfd885a1555d3834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 6272, 784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 8, 16, 49, 28, 28], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_4e6f52a10167ea3ad440e5912e662ad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 128, 28, 28], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_dab4dbb5d8d22009dd2650872907f47c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47173982858657837, 0.05311083793640137, 0.0470941998064518, 0.4884915351867676, 0.03961705043911934, 0.38735660910606384], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a0203d83cbd15989205660ee4f23d7c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 1, 7, 1, 7, 768], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_00cf3758e42cafe6006e258f6984d9df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 2304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 1, 49, 3, 24, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_6c2381b1dd9e9fc7597461a313d5926a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.168808251619339, 0.3926023244857788, 0.2295892834663391, 0.05097154155373573, 0.4849083423614502, 0.3777158856391907, 0.204880490899086, 0.34341758489608765, 0.16691364347934723, 0.10739654302597046, 0.2379564344882965, 0.08161421120166779], dtype='float32').reshape([12]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_772f42c160f86710ea5beb9568acd72a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 197, 3, 3, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_b59abea675318d3f7161a561e4581fc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 3, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 197, 192], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_db864debf850d39fdc8d03018e30851b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 16, 128, 256], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_4132d39f8dbfbe99f28e921876644816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 2, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 32, 128, 256], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5e3459691beccedab327e2bcf51a3a76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.018095821142196655, 0.2927481532096863, 0.48479756712913513, 0.488025039434433, 0.3335430920124054, 0.30857333540916443, 0.30590277910232544, 0.10617434978485107, 0.48340657353401184, 0.007441363297402859, 0.14749151468276978, 0.38283225893974304, 0.4328418970108032, 0.019393187016248703, 0.29819273948669434, 0.4008278548717499, 0.07727839797735214, 0.3019786477088928, 0.26260170340538025, 0.23957546055316925], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_be661e87e2b806f165a2f4b608d36819(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34045788645744324, 0.4461449682712555, 0.10263920575380325, 0.03984218090772629, 0.41042324900627136, 0.45167651772499084, 0.2664346396923065, 0.18790684640407562, 0.3772982060909271, 0.4684174656867981, 0.31876373291015625], dtype='float32').reshape([11]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e458380f2fe6841336df0e6b58b9fd80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([44], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0b1acb06b0664b5c49b3197f81a13f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_67885294bc04c397eaa071255384cd0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5bc320b26e1dd96923a1d619e75f258f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([400], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3723c9b72f82074ba6e33d843517e7c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 65536, 1, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ce7a405ad8b698e1a4bf0effd17a02ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 65536], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 32, 128, 512], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0ba04333f74ae611255cb6a244fe6822(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 32, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e58284ff04008d36628ad8135b1c8952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 1, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_9c82666dd2d7e2b086374da747710279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 65536, 32], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_48996548633eeb3d663bd81983882a21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([300, 256, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_886f76c8d62b5d29d0bba4ce699fc66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 300, 256], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d84a6d9e2384ff44cbcafff53bd52737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10461319237947464, 0.47418344020843506, 0.33790886402130127, 0.16025561094284058], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f6119ae77450f436b575dc9c64f51f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24057526886463165], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f93340ab48f526b1af83686981380919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 640], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 64, 8, 80], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ab39cafbf3486cc0fd8a562a3443590e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3246871531009674, 0.1638229489326477, 0.08918765187263489, 0.40535348653793335, 0.2557678520679474, 0.047975167632102966, 0.278403103351593, 0.41040441393852234, 0.48340851068496704, 0.17254047095775604, 0.222539022564888, 0.13782460987567902, 0.37139827013015747, 0.015496781095862389], dtype='float32').reshape([14]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a63c57602a1162b7aa44a4a5a3d89db0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_624fbd2849711c250b268a75674024e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10937570035457611, 0.4332655668258667, 0.2241179645061493, 0.3823685050010681, 0.1491249054670334, 0.489660382270813, 0.3012855648994446, 0.29865512251853943, 0.014166360720992088, 0.1973828226327896, 0.42504096031188965, 0.0894073024392128, 0.1883188635110855, 0.002881377935409546, 0.06808261573314667], dtype='float32').reshape([15]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8dccc8bc5773bc4065087b4e7bb99306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9ac8aee613482bdb6c9565418448e562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([8136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9ac8aee613482bdb6c9565418448e562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([8136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d900ecf63909826558023961b599b126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([138312], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4, 17], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_9ac8aee613482bdb6c9565418448e562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([8136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2b6f339bdc99694eb513483c6387e720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 4, 7, 4, 7, 192], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_499404e4daabc3b6f9a0e17f0e4041c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 16, 49, 3, 6, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_a4fe737c29f2de0610de8192d9e1d805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_229c49783a796ddc80521df8eb367bec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 16, 32, 160], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_524727cd00f23fb6eab95854ee357979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 14, 14, 384], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0baeab18cb58ef3fb0e44b31bdac67b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([8, 288, 8, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 288, 192], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e53314b10bb5e22b29665515e71cac24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([8, 288, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 1, 2, 24, 12, 192], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_ea7804d5ff00de627630afbb9f0a6c40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 24, 2, 12, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 24, 24, 192], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a4fe737c29f2de0610de8192d9e1d805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a4fe737c29f2de0610de8192d9e1d805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f6eabd80fd5ce69e13dee83d9de8b6a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 14, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 2, 72, 14, 25], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_fb179293eddcfc76251a785921c9d0da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 2, 14, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 144, 14, 25], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_84ece84c9ca4a847b8b015a64f4ec5eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 3136, 3, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_42008e42c0a564bc9128087afff0948a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 3136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 96, 56, 56], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c133d6039a6fd336eedea7099e9c4ce4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 96, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_9887b54065bff26b06e48385f02fa3a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 49, 2, 3, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8676fc8188c783c45ad6d1a6df9cf621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 384, 14, 14], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5eeda322c70054171f898924c87e8ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([64, 512, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 8, 8, 128, 4, 16], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_ba3b17bc15c9efb4c436d78c5b0a6d0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 16, 128, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 512, 8, 8], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1f4ea02f4f9f022aea3ddbd4dd9c16a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33217939734458923, 0.2088770717382431], dtype='float32').reshape([2]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cd6ceac712f80e55fab60d5ed3abf9ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10512582957744598, 0.26969438791275024], dtype='float32').reshape([2]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4789c149e53f6641b8df684101233e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27102673053741455], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c0156cfed6741fd3941c393340f4d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f80261241dca370c98381bc30b86e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_131e949067d872334214917a5607282c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 56, 56, 96], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d1d8acc99a522eb61ea3f691c0c9b37b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03555715084075928, 0.41032981872558594, 0.35534191131591797, 0.08809032291173935, 0.06647709012031555, 0.28557687997817993, 0.2918945848941803, 0.41252991557121277, 0.303602933883667, 0.39854326844215393, 0.318382203578949, 0.33693036437034607, 0.08150046318769455, 0.33481070399284363, 0.33833444118499756, 0.14796455204486847, 0.36273959279060364, 0.1442449986934662, 0.1844324916601181, 0.10331367701292038], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d2fc1ce4528452e5dc6fe8e2c85da6d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([18668], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d2fc1ce4528452e5dc6fe8e2c85da6d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([18668], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_62fa06c0d8e933883dcd77cca69ae583(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([317356], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4, 17], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_d2fc1ce4528452e5dc6fe8e2c85da6d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([18668], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_459c14656e433fc8693ce432250bb635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.452399343252182, 0.25411558151245117, 0.186106875538826, 0.23583035171031952, 0.4823371171951294, 0.2429245561361313, 0.11906544864177704, 0.21629804372787476, 0.327368825674057, 0.2696342170238495, 0.24014686048030853, 0.2506621479988098, 0.3724493086338043, 0.3788430988788605, 0.3108941614627838, 0.4442995488643646, 0.000622588733676821], dtype='float32').reshape([17]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7ef6c51dd626822425a0a7a84bc8b703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 3, 4, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_18884e65102af273b2f5c32e2b945d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 4, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 128], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a4fe737c29f2de0610de8192d9e1d805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3d3a82143a2e5de7fe35559525684ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3503669500350952, 0.14494989812374115, 0.3783012926578522, 0.12024679780006409, 0.09634096920490265, 0.08221487700939178], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9638bcad3927251b1b2258aae5e91e42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([4704], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([6, 28, 28], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1f236bc4a21bd55e9054f5b3a8be0c85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12026521563529968, 0.2722843587398529, 0.33296436071395874, 0.1062338799238205, 0.4873442053794861, 0.31150946021080017, 0.35609719157218933, 0.4547961950302124, 0.4935852289199829, 0.33908411860466003, 0.16300922632217407, 0.42115628719329834, 0.0370887815952301, 0.30867525935173035, 0.4606568217277527, 0.44711804389953613, 0.2791314423084259, 0.04940961301326752, 0.155556321144104, 0.10583055019378662, 0.10259288549423218, 0.24823671579360962, 0.04379118233919144, 0.004167107865214348, 0.11252257227897644, 0.4044469892978668, 0.3088270425796509], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_85e8663c4b36aacb75e92536f9bdb660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e4cff88a6ff746f66a8269c8b14bf4f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([4208], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e4cff88a6ff746f66a8269c8b14bf4f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([4208], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5ca5ba3a0b5f8b45c6768d7bcce2ef45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([71536], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4, 17], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e4cff88a6ff746f66a8269c8b14bf4f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([4208], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_aef3050cf1b71ce3375b986ba3e2c6cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.047258492559194565, 0.27776533365249634, 0.03029080294072628, 0.21338896453380585, 0.03320344164967537, 0.007270867470651865, 0.06549911201000214, 0.10299830138683319, 0.4138154685497284, 0.04399467259645462, 0.06624656915664673, 0.20735305547714233, 0.3706805408000946, 0.14700183272361755, 0.15227660536766052, 0.23536641895771027, 0.3211774230003357, 0.09582307934761047, 0.22759445011615753, 0.1748839169740677, 0.05285251513123512, 0.4828563928604126, 0.16188286244869232, 0.49291905760765076], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1a6bfdca91d98b9dda797712ecade59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.031434327363967896, 0.028465917333960533, 0.4268665015697479, 0.0047273654490709305, 0.2398131787776947, 0.015620541758835316, 0.2842951714992523, 0.2755797207355499, 0.2227810025215149, 0.1865205615758896, 0.42923688888549805, 0.017299212515354156, 0.2419130504131317, 0.2982611656188965, 0.4690977931022644, 0.23285794258117676, 0.20459426939487457, 0.27397552132606506, 0.08335361629724503, 0.3266809284687042, 0.46778255701065063, 0.3122619390487671, 0.4286305010318756, 0.053278759121894836], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6078b09706e998f6505beda1abf4cd1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03030535764992237, 0.2509848475456238, 0.35065943002700806, 0.3615802526473999, 0.3483794033527374, 0.4121735394001007, 0.35178643465042114, 0.2682470679283142, 0.08162184059619904, 0.029156194999814034, 0.38795262575149536, 0.0352461114525795, 0.015652192756533623, 0.29451850056648254, 0.32424500584602356, 0.20823271572589874, 0.028011249378323555, 0.42809033393859863, 0.332924485206604, 0.0692853331565857, 0.1543518602848053, 0.0025105425156652927, 0.08021824061870575, 0.47115543484687805], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ad71536a7fc7b6a92f90d9b4a87bfd2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3394601047039032, 0.4115154445171356, 0.4541627764701843, 0.3653869032859802, 0.14694564044475555, 0.09159527719020844, 0.021768806502223015, 0.019286824390292168, 0.1259915977716446, 0.11486314237117767, 0.015221644192934036, 0.12770850956439972, 0.0025629806332290173, 0.4515746533870697, 0.031507041305303574, 0.0659373551607132, 0.04686928912997246, 0.13015282154083252, 0.4338769018650055, 0.4802113473415375, 0.02977254055440426, 0.14390160143375397, 0.11162090301513672, 0.3438325524330139], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b7f9d94620f4305fbb7bd18c6b1c6db0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2739105820655823, 0.24946492910385132, 0.1715574562549591, 0.487457811832428, 0.40567970275878906, 0.25944405794143677], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cfe570d099149aaf11a89e77d0af2251(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42763781547546387, 0.29627394676208496, 0.2104211300611496, 0.0948689803481102, 0.4824773073196411, 0.048598721623420715, 0.2829347252845764, 0.0009231431758962572, 0.13366210460662842, 0.47216030955314636, 0.3070950508117676, 0.4739173352718353, 0.032722458243370056, 0.3119249641895294, 0.1084766760468483, 0.39215242862701416, 0.16332454979419708, 0.23771484196186066, 0.40048927068710327, 0.19026009738445282, 0.1930728405714035, 0.09304925054311752, 0.2473549246788025, 0.052767835557460785], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_987ba302a659c69a318d33eb23dd9aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3725551962852478, 0.4389555752277374, 0.34317904710769653, 0.29388928413391113, 0.04986639320850372, 0.04697471112012863], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b3d655f5a3686395fae09fe477669378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23891767859458923, 0.3738255500793457, 0.4852713346481323, 0.125051349401474, 0.4336668848991394, 0.05884374678134918, 0.10693471133708954, 0.36345916986465454, 0.08292428404092789, 0.14499841630458832, 0.07481024414300919, 0.11827903985977173, 0.2534327805042267, 0.2579708993434906, 0.06978116184473038, 0.43650734424591064, 0.2660924196243286, 0.038182832300662994, 0.3479653298854828, 0.4692835807800293, 0.2737882733345032, 0.1709263026714325, 0.12146316468715668, 0.14367635548114777], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6a4454039dd87f370dfa0c7c8f2fd386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08321098983287811, 0.12371540814638138, 0.056202471256256104, 0.3115832805633545, 0.0983789712190628, 0.06569509208202362], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ac9bfb6d309e363afbe409b6e6b3a58a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06944896280765533, 0.16716063022613525, 0.48116335272789, 0.1568867266178131, 0.2876695394515991, 0.019057663157582283, 0.4829829931259155, 0.06431084871292114, 0.17748364806175232, 0.08289823681116104, 0.15830378234386444, 0.38012048602104187, 0.3013538122177124, 0.42249494791030884, 0.31456589698791504, 0.01922636479139328, 0.051463980227708817, 0.3417123258113861, 0.4352193772792816, 0.19191469252109528, 0.11644327640533447, 0.4984947144985199, 0.14757171273231506, 0.49113941192626953], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a8a5290d8abcb0c1d3b0b02d49a12d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03367343172430992, 0.4177984595298767, 0.152892604470253, 0.45381399989128113, 0.15641658008098602, 0.49384093284606934], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bbb21e46a1737bdedffca9df40188210(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48136118054389954, 0.2821231186389923, 0.3948209881782532, 0.4913710951805115, 0.026751205325126648, 0.47853612899780273, 0.39972954988479614, 0.42198893427848816, 0.3319377601146698, 0.3279159963130951, 0.40910863876342773, 0.24989847838878632, 0.4002339243888855, 0.47013622522354126, 0.04404428228735924, 0.1885397881269455, 0.2868679165840149, 0.3428848683834076, 0.26419463753700256, 0.006731228902935982, 0.0829000398516655, 0.1268075406551361, 0.2004116028547287, 0.03437739983201027], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cd664a662fbbea631f3d56c19bbd54c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_df7143c696ffe1946e6762afedef9dfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 32768, 1, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7c0440355e1d4f46acb0277a6e5110df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 64, 128, 256], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_68c590797e4c26abf52d59db7afd1779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 64, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a32be331ccba8c42c6958c98703be692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 1, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_80772ce342e5a56e23b4c7c105ad579d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 32768, 64], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5442f0f2b647eeab7f0512395152951b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 49, 24, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_28e495d1af865cff25f7e13cfba0df28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1536], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 49, 2, 24, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b704e8a2057ee823f83fcb49cedfa20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_37dd8ade427fac57583faed4890971dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b5e24984e01b5e5f31996ff6dbfd8b45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2251991480588913, 0.45815059542655945, 0.39652714133262634, 0.4037114083766937, 0.07742254436016083, 0.32232236862182617, 0.3247644603252411, 0.39447760581970215, 0.052055876702070236, 0.025493254885077477, 0.06339961290359497, 0.052468352019786835, 0.4660857915878296, 0.26535409688949585, 0.04680509492754936, 0.2601557672023773, 0.39274337887763977, 0.4090096950531006, 0.23782426118850708, 0.35129156708717346, 0.32676827907562256, 0.04492872208356857, 0.46866151690483093, 0.1592639833688736, 0.3819335699081421, 0.22358503937721252, 0.2710113525390625], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f7f62f2f0f020f9e6a466633eb6ff475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5da7b4cb2bce4ffb80abfdd4c841764d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 16, 64, 150], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4a413168a1978c6812c196da229a44d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20218968391418457, 0.2763853073120117, 0.3523274064064026, 0.179668128490448, 0.09864689409732819, 0.3334880769252777], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_71f0c420838e2f68e1748f4bd5434735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 3, 2, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_701db45360226e2719decfbfea469937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 2, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 64], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_65dab9ef3923dafd4d8d5abede2a13dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20146335661411285, 0.41295188665390015, 0.17795665562152863, 0.16160713136196136, 0.09434108436107635, 0.04255187138915062, 0.11241578310728073, 0.16893355548381805, 0.09889499098062515, 0.33937913179397583, 0.24688565731048584, 0.16702178120613098, 0.3854455351829529, 0.11929043382406235, 0.3780512809753418, 0.270608514547348, 0.10461796820163727, 0.3338450491428375, 0.42974257469177246, 0.3978571891784668, 0.1618420034646988, 0.37082839012145996, 0.04574419558048248, 0.11191573739051819, 0.011609278619289398, 0.2060665339231491, 0.26939108967781067], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_85e8663c4b36aacb75e92536f9bdb660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_59658164a2532bfdf6cf68a24960ce73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 9261, 4, 17], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_aec1d9b027e3af13057a561475abe80a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3891236186027527, 0.14547526836395264, 0.3994063138961792, 0.10415982455015182, 0.28679898381233215, 0.37694793939590454, 0.3399457335472107, 0.1065632700920105, 0.4667667746543884, 0.07119045406579971, 0.46810483932495117, 0.3584262430667877, 0.3103014826774597, 0.11798892915248871, 0.47606387734413147, 0.3102288246154785, 0.10136575251817703, 0.42436671257019043, 0.2148064225912094, 0.47864091396331787, 0.2604401111602783, 0.40532487630844116, 0.11874133348464966, 0.42680633068084717, 0.37863531708717346, 0.12482216209173203, 0.3423089385032654], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_85e8663c4b36aacb75e92536f9bdb660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_053af562f8484f283ac885248882ae9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 16, 16, 16], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_65922d0f9b4e820b51d0b0355b600c82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([16, 784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 16, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cf50a242b12d20e39a4343b154112f58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21596314013004303, 0.13437269628047943, 0.433734655380249, 0.34139400720596313, 0.4562983512878418, 0.03436760604381561, 0.27477240562438965, 0.0543794184923172, 0.33159902691841125, 0.39123794436454773, 0.060426585376262665, 0.17691293358802795, 0.08249394595623016, 0.1508890688419342, 0.021088356152176857, 0.3364042341709137, 0.34851008653640747, 0.4569130539894104, 0.4845907986164093, 0.41989609599113464, 0.05467788130044937, 0.4737434685230255, 0.033862221986055374, 0.45528605580329895, 0.26711511611938477, 0.19842588901519775, 0.3446415066719055, 0.4799250066280365, 0.07558789849281311, 0.32773444056510925], dtype='float32').reshape([30]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_50c6c03c87b49b725f20fe19b4e5f3b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23553751409053802, 0.047110460698604584, 0.306405633687973, 0.41685017943382263, 0.23628130555152893, 0.25448161363601685, 0.08752992749214172, 0.30875363945961, 0.1443181186914444, 0.09441345185041428, 0.2149564027786255, 0.026198627427220345, 0.470008909702301, 0.09538552910089493, 0.15369534492492676, 0.2276269793510437, 0.4123591184616089, 0.26476481556892395, 0.1051313504576683, 0.3383327126502991, 0.38706451654434204, 0.33593088388442993, 0.365531861782074, 0.1869570016860962, 0.4793265759944916, 0.2652244567871094, 0.1641613095998764], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_85e8663c4b36aacb75e92536f9bdb660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_459827ff6f66eee9bc84a0a117613746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2595360577106476, 0.2638426423072815, 0.2675228416919708, 0.2569276690483093, 0.415460467338562, 0.4887153208255768, 0.022583158686757088, 0.3260241448879242, 0.10097033530473709, 0.27124500274658203, 0.3638888895511627, 0.08002448827028275, 0.12559328973293304, 0.07940618693828583, 0.2881127595901489, 0.276809424161911, 0.44217342138290405, 0.2647753953933716, 0.13132625818252563, 0.25325942039489746, 0.10990816354751587, 0.08995982259511948, 0.15945810079574585, 0.2492533028125763, 0.22434164583683014, 0.03997206315398216, 0.4176250100135803, 0.36801403760910034, 0.4175297021865845, 0.4574833810329437], dtype='float32').reshape([30]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6b016b1ac988abf889f48fec4323bcbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1852516382932663, 0.3793354630470276, 0.24261508882045746, 0.4227461814880371], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a2556311a5f4d1eecdca3718b72ad7d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09735648334026337], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_84056e78911b378284f3f81a67641db6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2944941222667694, 0.3053193688392639, 0.2151702642440796, 0.044051844626665115, 0.43019893765449524, 0.013311056420207024, 0.09974949806928635, 0.15193189680576324, 0.2166559100151062, 0.31936943531036377, 0.34587815403938293, 0.4404135048389435, 0.1799142211675644, 0.09006453305482864, 0.4967118203639984, 0.22567328810691833, 0.39930763840675354, 0.03425982967019081, 0.02978592924773693, 0.3281196355819702, 0.33778414130210876, 0.33493268489837646, 0.03154968470335007, 0.1622592657804489, 0.19737356901168823, 0.30155953764915466, 0.3705887198448181, 0.23753014206886292, 0.3168289065361023, 0.4820905327796936], dtype='float32').reshape([30]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1a52a7d6c357d332ce26ad44521544ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 197, 2, 6, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_72173ab5bc40f947f9e2313cff94c9a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 197, 6, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2d025537a0b06059c49c140194ad4421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d08255f9de23ebb3f061ef4a588e620(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_717a2ef1aab87cd94a891c5fae045402(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 128, 4, 25], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_475491f1d237d011f2b1a05b2f26736b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 21760], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 96, 136, 160], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_871e7ed58ae24dd205223e8bab4c5cbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([240, 96, 4, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 96, 96], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_37431aa7674aafb85f36f16ed818762e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([240, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 1, 24, 48, 2, 96], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_25e14d9e1ebaf44938698ae9dedef4a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 48, 24, 2, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 48, 48, 96], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ca5aa789752502dd1c15f4a41a14d367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8a92b7f362e14d4e918031235743d060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 32, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 144, 768], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_1cbd062d6aea9edd1d8b97d75ee18b61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 1, 1, 12, 12, 768], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_44e7b0866f95f3eab8255be4d755b7c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 12, 1, 12, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 12, 12, 768], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_39c6d15321088eec074af2f7e62a775e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1568], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1c8561c110dec7fd4d0e9d1fd5b28924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1568, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 32, 49, 7, 7], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_404abdb05bc7e174198a046c369019f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 25088, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 32, 16, 49, 7, 7], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_56554c8d9fe96c18d78f9e9d9495cc42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 512, 7, 7], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6f474456a9bb840ce0671ac08e9ecab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7f13a12635ac243138c909e8e3b8f0d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([68, 68, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8dbfc93a894aff6cec5d622dac5cf395(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([34, 34, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bfdacd65165f54513143b021ccee98b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([17, 17, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7f13a12635ac243138c909e8e3b8f0d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([68, 68, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8dbfc93a894aff6cec5d622dac5cf395(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([34, 34, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bfdacd65165f54513143b021ccee98b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([17, 17, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_958d315129b9fc8482f335d21d85ab61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2386901080608368, 0.03617676720023155, 0.22566987574100494, 0.38778156042099, 0.4396834373474121, 0.22753463685512543, 0.2621132731437683, 0.39386188983917236, 0.3504861295223236, 0.22841818630695343, 0.24599675834178925, 0.008698318153619766, 0.3863283097743988, 0.36733317375183105, 0.1329926997423172, 0.39610329270362854, 0.4708232581615448, 0.10890984535217285, 0.3398969769477844], dtype='float32').reshape([19]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e0b7efd60fa9e55697219e588953fdc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18207412958145142, 0.20705540478229523, 0.04756245017051697, 0.14409857988357544, 0.09395479410886765, 0.2180514931678772, 0.2956453859806061, 0.3639739751815796, 0.20787717401981354, 0.4486352801322937, 0.4410051107406616, 0.44928544759750366, 0.4646989703178406, 0.19928213953971863, 0.41427281498908997, 0.3979845345020294, 0.046477872878313065, 0.15579171478748322, 0.4762071371078491], dtype='float32').reshape([19]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7dab697a7b9d3b17895f8d6b0ebf1150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2426489144563675, 0.496039479970932, 0.33659306168556213, 0.4481460452079773, 0.08138708770275116, 0.23930375277996063, 0.26480093598365784, 0.4485655725002289, 0.3295544385910034, 0.30528524518013, 0.04899377375841141, 0.08693577349185944, 0.4148196578025818, 0.21409323811531067, 0.21144162118434906, 0.23832371830940247, 0.36082011461257935, 0.09621884673833847, 0.35962027311325073], dtype='float32').reshape([19]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5dac8d93c6872f60a80db656baeff4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([84864, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_35b76a6a539241ef1a71ec09f15ccdc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([21216, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3c6445ac28ec3b2cd10dced8e1d2d991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([5304, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ecd05fd8531c040f5e4b451dfd959432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([1326, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9ef0bc657dc56f9add590b1f52444ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([351, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_38b42bf3cf2e375aa962a6b30811e51b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 136, 208, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_4e3eac56d72535c5234451b8780d1a2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 104, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b623464b5e921898000e083b41349e72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 34, 52, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_fcd787e481e36732f115c86197de449e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 26, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_62227069f45f12f525c7a932d6789675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 13, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8fb97f2e09d759b44d59bc3516eef104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 136, 208, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_91bd032aa73d5db00bd1253b3ae25308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 104, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_074bf876facb083fc086ca9d7b8eca3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 34, 52, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_4a716b2ed46c4b72d469931b24d87212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 26, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_10deff7dfd75656c84805ccbd1eed574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 13, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_1090a244460bd9019ea023a58b21e79d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.36567196249961853, 0.06277179718017578, 0.36737382411956787, 0.09195312112569809, 0.07496287673711777, 0.38207003474235535, 0.07352693378925323, 0.27364858984947205, 0.09849617630243301, 0.09296906739473343, 0.2122330367565155, 0.02770385704934597, 0.28670039772987366, 0.4851173758506775, 0.4216403663158417, 0.37599337100982666, 0.16203193366527557, 0.4721679389476776, 0.0316360667347908, 0.3612683117389679, 0.06854290515184402, 0.3050507605075836, 0.13655903935432434, 0.04856431856751442, 0.21685422956943512, 0.005126427859067917, 0.44502851366996765, 0.40384674072265625, 0.18095649778842926, 0.38175398111343384], dtype='float32').reshape([30]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8396af63fa7f79de582f076e00a06985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 28, 28, 192], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_30c0f18074348673574a8703f9d86b92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3038213551044464, 0.23970966041088104, 0.4657522439956665, 0.36774536967277527, 0.06891488283872604, 0.39472946524620056, 0.3875124752521515, 0.43949759006500244, 0.04184189811348915, 0.31851881742477417, 0.36659204959869385, 0.06241467967629433], dtype='float32').reshape([12]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_686e383e15a67641022e6325124aaa73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 2, 7, 2, 7, 384], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_31d8bb07d21e80ee441322f58a3742c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 1152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 4, 49, 3, 12, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_5c9cc55cb2eb60e1e3a0a10bb97f77e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 8192, 2, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0becbdf458d00e3901d03f6b82169fbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 64, 128], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_65af31d73c70cd41b8c96bf16079e864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2d3486cafb0a53340f7f41bd8204aedf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 2, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_c285f356d9d3f2aa0d2345afc9677530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 2, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 8192, 128], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5d090e3a79f7fca556b4dd81b647bdf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12826396524906158, 0.22494348883628845, 0.05054886266589165, 0.24675697088241577, 0.11677438765764236, 0.03578204661607742, 0.41347354650497437, 0.33050063252449036, 0.18624424934387207, 0.3405345380306244, 0.3917767405509949, 0.45712071657180786, 0.4447803497314453, 0.19758079946041107, 0.2562739849090576, 0.47609636187553406, 0.009760121814906597, 0.4921445846557617, 0.47976943850517273, 0.03359748795628548, 0.03872550651431084, 0.04661550745368004, 0.114780955016613, 0.3540026545524597, 0.36918962001800537, 0.2562340795993805, 0.029675107449293137], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_85e8663c4b36aacb75e92536f9bdb660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c2d6279151787c36ec263aefa3e83c4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10972460359334946, 0.045949310064315796, 0.018521210178732872, 0.0936998501420021, 0.11876701563596725, 0.2974971532821655, 0.40804323554039, 0.36584019660949707, 0.2860509753227234, 0.2188769429922104, 0.02849111519753933, 0.178543359041214], dtype='float32').reshape([12]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_656ca647506987cf60a981c21f7b7bdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1561281681060791, 0.48436805605888367, 0.038039494305849075, 0.0923214852809906, 0.49722373485565186, 0.14648212492465973, 0.48690176010131836, 0.47466573119163513, 0.4179636240005493, 0.2354658842086792, 0.4592534601688385, 0.33007246255874634, 0.414271742105484, 0.44682982563972473, 0.4304667115211487, 0.4115184247493744, 0.45717519521713257, 0.42003777623176575, 0.023680562153458595, 0.08714810013771057, 0.3907470107078552, 0.4870944321155548, 0.032428815960884094, 0.28629010915756226, 0.4590838551521301, 0.010420478880405426, 0.1576608419418335, 0.3278587758541107], dtype='float32').reshape([28]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_46220188c52a9bf37c6215199e8ba514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20686180889606476, 0.2887507379055023, 0.4501131474971771, 0.3195890188217163, 0.42185506224632263, 0.49884042143821716, 0.002723176497966051, 0.3133925795555115, 0.3551175594329834, 0.3718971014022827], dtype='float32').reshape([10]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_940b65524433657086eff3256fa78c3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f10b91eeaa45ba5106bd12c92acf2d26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_4ee3b807c30c71482b85c15453249e3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_89234e48501fe6fe1f7e34c5e64b3a1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([16, 16, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_05b6d101a6c0d72ad41a9c36ef2dc778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([8, 8, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8c10e541823a5a53c11af72eb7f5ddd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.36022669076919556, 0.3078524172306061, 0.3102732002735138, 0.1274881362915039, 0.3704889118671417, 0.4765748679637909, 0.0032886615954339504, 0.13248226046562195, 0.48888346552848816, 0.03138444200158119, 0.05153635889291763, 0.2607216536998749, 0.25766485929489136, 0.15790516138076782, 0.3261634111404419, 0.06399227678775787, 0.046346791088581085, 0.11954301595687866, 0.01980046182870865, 0.014799871481955051, 0.19580857455730438, 0.36806175112724304, 0.07330067455768585, 0.3947322368621826, 0.13428373634815216], dtype='float32').reshape([25]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_67885294bc04c397eaa071255384cd0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f11f112b8286593dfd01fd61640a2437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2048, 5, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_652a250bb074f5951353b9e804b55fbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 2048], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 320, 32, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b4d48f961bdd6507ed527888faea93b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f9b2842dfeb4871418a139598f4b1998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 320, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_9de7b0f60e0726ca9d88269bc70656de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 640], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 5, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_09d7503de67eedb7c74c65e8689e4eef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 5, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2048, 320], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1d35570dadea980ad61998cdd7c6f5d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0b1acb06b0664b5c49b3197f81a13f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8dccc8bc5773bc4065087b4e7bb99306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0f37cad57a8e96307e4e89f46ffef52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2100, 4, 17], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_68f0d3065e11412f55d1465379424a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 64, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_68f0d3065e11412f55d1465379424a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 64, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b4ac73d7610f568d2634260d318cfaaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 512, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_d3431672b6352d32a3c99ea23930fc26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 512, 64, 128], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6831855fd545ff434c2fa0b7c2603739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1709495484828949, 0.24623730778694153, 0.31642094254493713, 0.19901494681835175, 0.4967736005783081, 0.12761183083057404, 0.08886690437793732, 0.12540821731090546, 0.25236669182777405, 0.15741343796253204, 0.2942917048931122, 0.15931406617164612, 0.22361312806606293, 0.0666620209813118, 0.116729736328125, 0.025919318199157715, 0.4581216871738434, 0.41823098063468933, 0.35308775305747986, 0.057389020919799805, 0.1340121626853943], dtype='float32').reshape([21]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_90865c95e25d1e3117bc3aa012eb32e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1729038506746292, 0.18732747435569763, 0.26402392983436584, 0.1891935020685196, 0.1288832575082779, 0.13410717248916626, 0.3428444564342499, 0.022292818874120712, 0.4665474593639374, 0.44318586587905884, 0.4731910228729248, 0.32583189010620117, 0.15058426558971405, 0.3278496265411377, 0.45202863216400146, 0.3516013026237488, 0.14963693916797638, 0.47564297914505005, 0.21714189648628235, 0.19560495018959045, 0.014795549213886261], dtype='float32').reshape([21]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5a19262984854ce5935b0697bbbd99d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24392203986644745, 0.32638391852378845, 0.29007917642593384, 0.16348667442798615, 0.13434281945228577, 0.0387825109064579, 0.009858394972980022, 0.13167612254619598, 0.35689452290534973, 0.2900632917881012, 0.3181535303592682, 0.3350951671600342, 0.12291521579027176, 0.1844567209482193, 0.15458311140537262, 0.2961300015449524, 0.14721038937568665, 0.35147902369499207, 0.3781965672969818, 0.17529937624931335, 0.19997772574424744], dtype='float32').reshape([21]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1d35570dadea980ad61998cdd7c6f5d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ade97f0a5d8245b042bc8ed50c954c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.031210971996188164, 0.44997668266296387], dtype='float32').reshape([2]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_90f1cf205ee4ee76dd76bad5753625e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11430265009403229, 0.29206782579421997], dtype='float32').reshape([2]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ca12aa9a921a4b4427384f718d1fc00d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07972206175327301], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d1ab532e96ea87b051a4682bdb73d3aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 12, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f9bbb317a048eb324b0f527fb48c88dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 144, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_524727cd00f23fb6eab95854ee357979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 14, 14, 384], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_dad364fd958c2949ba32af8acea2b86a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1271533966064453, 0.3956158757209778, 0.0008763488731347024, 0.4500281810760498], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_28219a5a32202bdf0c4d299a107583ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10698025673627853], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_918e13c281a13bd80608d45c9e0ac990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 14, 14, 384], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_669f967f2e23c1a676884c36022a5f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 1, 7, 1, 7, 768], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_469080bbaee58dc9f7576fcb882cc6b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 2304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 1, 49, 3, 24, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_567d743ae088dac5db93d4ac801307ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 768, 7, 7], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8dccc8bc5773bc4065087b4e7bb99306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3ebf5d5e9a247ecee59bb90231b0c141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3ebf5d5e9a247ecee59bb90231b0c141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3ebf5d5e9a247ecee59bb90231b0c141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3ebf5d5e9a247ecee59bb90231b0c141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3ebf5d5e9a247ecee59bb90231b0c141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3ebf5d5e9a247ecee59bb90231b0c141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3ebf5d5e9a247ecee59bb90231b0c141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3ebf5d5e9a247ecee59bb90231b0c141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 784], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_39533a9d34962f9c4d60519d9a2b509b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4039015471935272, 0.2536815404891968, 0.4724632501602173, 0.26468098163604736, 0.19953736662864685, 0.1566523313522339, 0.18601025640964508, 0.14952905476093292, 0.46676716208457947, 0.29521310329437256, 0.25508591532707214, 0.1303054690361023, 0.07093783468008041, 0.4532754421234131, 0.40343934297561646, 0.22212950885295868, 0.18890605866909027, 0.14777134358882904], dtype='float32').reshape([18]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8dccc8bc5773bc4065087b4e7bb99306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9a7b39e9137dddde7624f0d2cdc5d5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 2, 7, 2, 7, 384], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_a3db33b717a764a26f815ee0728c4343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 1152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 4, 49, 3, 12, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_669f967f2e23c1a676884c36022a5f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 1, 7, 1, 7, 768], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_469080bbaee58dc9f7576fcb882cc6b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 2304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 1, 49, 3, 24, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_f3f0a13b0ec9d5c5346abae9118d039d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 2304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 1025, 3, 12, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_3afc9fc3c9a994ff298a526c65508f11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 12, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 1025, 768], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_376accb1e59e5b8bd501f627602c1425(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([9512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_376accb1e59e5b8bd501f627602c1425(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([9512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8152c97d17a719531fdac3b70cebbede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([161704], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4, 17], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_376accb1e59e5b8bd501f627602c1425(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([9512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a7f7b19adf313da7b8bec51e56ea6f7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3614782691001892, 0.4247654378414154, 0.2354958951473236, 0.4199094772338867, 0.4829670190811157, 0.45896461606025696, 0.19075556099414825, 0.30672189593315125, 0.11279232800006866, 0.21992334723472595, 0.4294649660587311, 0.29170092940330505, 0.021494122222065926, 0.2297290563583374, 0.4200938642024994, 0.352750688791275, 0.003987254109233618, 0.3243481516838074, 0.23440203070640564, 0.0688839852809906, 0.48129522800445557, 0.024555284529924393, 0.34763824939727783, 0.3509567975997925, 0.17522983253002167, 0.2606576085090637, 0.25096866488456726], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f7f62f2f0f020f9e6a466633eb6ff475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8dccc8bc5773bc4065087b4e7bb99306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d6cf2ac44eb3292e29b8127bd85211f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([44, 288, 8, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 288, 192], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_50c2292bba7214c90d03e27fb464063b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([44, 288, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 1, 2, 24, 12, 192], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_e81142df1203486af81b32b20a59875c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 24, 2, 12, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 24, 24, 192], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9a7b39e9137dddde7624f0d2cdc5d5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 2, 7, 2, 7, 384], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_a3db33b717a764a26f815ee0728c4343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 1152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 4, 49, 3, 12, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_0e03424f762205714f43e39bd7b31dcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33039358258247375, 0.35414132475852966, 0.051127925515174866, 0.2933516800403595, 0.048918467015028, 0.11756949871778488, 0.47161611914634705, 0.12734025716781616, 0.07302495837211609, 0.01591285690665245], dtype='float32').reshape([10]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a0e868904eb81bff33750804fee29be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([12420], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a0e868904eb81bff33750804fee29be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([12420], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_66bec75b2b5313031547a922ec39e66d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([211140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4, 17], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a0e868904eb81bff33750804fee29be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([12420], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3d5d310eed30aed0e9a630035b2aaa48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([15328], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3d5d310eed30aed0e9a630035b2aaa48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([15328], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_da438f95af70efe3b3ac9a34268260d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([260576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4, 17], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3d5d310eed30aed0e9a630035b2aaa48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([15328], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_12f2142378f5699021e379bc2dfad894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05648074299097061, 0.3060809373855591], dtype='float32').reshape([2]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e0b86f0f845960783d817a3d04d28a9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4199466109275818, 0.46029379963874817], dtype='float32').reshape([2]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_01778455c55afaa781b4711178fa4a10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23038315773010254], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_92f9363477bb01a340c583a36246db78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1201745942234993, 0.33536678552627563, 0.41566675901412964, 0.11899502575397491, 0.30160796642303467], dtype='float32').reshape([5]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c8e1fd2193867efd993c68aa42c0fd61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44550156593322754, 0.06368760764598846, 0.19570213556289673, 0.31796717643737793, 0.2168029248714447, 0.2489003837108612, 0.24345579743385315, 0.46153804659843445, 0.12765593826770782, 0.3889043629169464, 0.36925017833709717, 0.02241719886660576, 0.4907730221748352, 0.1052779033780098, 0.2385922223329544, 0.26012560725212097, 0.39151933789253235, 0.10009722411632538, 0.2711490988731384, 0.16047890484333038], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_654969e063f9680191edfa1289ffa723(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3293180763721466, 0.05654066801071167, 0.38562363386154175, 0.07279690355062485, 0.1605471968650818, 0.4452693462371826, 0.1350281685590744, 0.09596800059080124, 0.10942487418651581, 0.44108182191848755], dtype='float32').reshape([10]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c0156cfed6741fd3941c393340f4d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7cb16fe1f9c05951f09ac73e99139d87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4211239516735077, 0.41602030396461487, 0.3188924491405487, 0.44293439388275146, 0.4697565734386444, 0.2690028250217438, 0.15287557244300842, 0.060994766652584076, 0.08499234914779663, 0.08914633840322495, 0.14417323470115662, 0.13816682994365692, 0.2859761416912079, 0.3701353371143341, 0.4298166036605835, 0.15671171247959137, 0.033779386430978775, 0.14843153953552246, 0.4923192858695984, 0.20443366467952728], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c0156cfed6741fd3941c393340f4d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f80261241dca370c98381bc30b86e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_db1388a65a19770df872c92d63ea55d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 20, 128, 256], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_ca82a4da28970b48bd75e52ae8bfebe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 2, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 40, 128, 256], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1e2c7f966d81db07ffdd7e160cd0181b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 40, 64, 128], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_8c2e63af0d0bee98a4f0c30d350a56bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 2, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 80, 64, 128], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_430b978457976f24aac3b5bf75f4b1ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 80, 32, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_8831deb64d6d534db0e3b71b7a6c85f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 160, 32, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7e45064559fa780827a85ab092ad514a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 160, 16, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_ee00afbef1d38bc840dafb262ec9a1c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 2, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 320, 16, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2d025537a0b06059c49c140194ad4421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d08255f9de23ebb3f061ef4a588e620(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8afcac4171042362e2b41d97782c69f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([139392, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3a929403af8a1423214d0afd99e6da1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([34848, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9d7cb5dde6317e9487d24410ecfd6481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([8712, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4dcc60ee618cfe06fff655569c47917f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([2178, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cf63ced9c157e8b2a7532c348becd945(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([561, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_47c77d8d595097698fbb53426b93d4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 176, 264, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7d8aab9d6f18aad4553aea50b8532d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 88, 132, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_85300b4c002adf9938c9cafa14c44db8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 66, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2932449ca12a9c4437dc3185e7182ea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 22, 33, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_ff1b609f2619dd186b83dc79592ed347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 11, 17, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5301f07d8754be046c939f64ced59ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 176, 264, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_d4eedb08c59264888d4b4cb9b5032739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 88, 132, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_12fdfac20e142eff367441a52bf2c0f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 66, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_290a82df2aa3b1e7729236babaa6626d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 22, 33, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b5d3470ba6fc396fabedcfa22770742b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 11, 17, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_131e949067d872334214917a5607282c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 56, 56, 96], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b4d48f961bdd6507ed527888faea93b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8a580b528bf2ea4f31a0352a26b4b313(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08193186670541763, 0.2981763780117035, 0.4009934663772583, 0.3846578001976013, 0.4976460933685303, 0.40146908164024353, 0.1296776831150055, 0.1957193911075592, 0.06788989156484604, 0.16141870617866516, 0.4306678771972656, 0.33951863646507263, 0.47641250491142273, 0.2670423686504364, 0.005945517681539059, 0.20577797293663025, 0.47615665197372437, 0.31864210963249207, 0.1420571655035019, 0.19530317187309265, 0.37672409415245056, 0.22169151902198792, 0.22604215145111084, 0.2182777374982834, 0.05397915467619896, 0.4177415072917938, 0.3439372181892395], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d8c3ad013e5e88f8095e74d509cdf926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_29fb9c39eda326b391a064e451c579e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([100, 256, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e442ecd66d5901639c355073b02dc02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 100, 256], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b4ac73d7610f568d2634260d318cfaaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 512, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_d3431672b6352d32a3c99ea23930fc26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 512, 64, 128], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7f2ee97b3e4789bc72d2f9c1c2ec6be4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1024, 8, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e126d17d99dc3106c7b08af164b74981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 8, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_148ccf7f9b19aa4b5dc2d57c895e08b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1024, 512], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_9d770d7bcb4b2cc9647f74b6ba94e982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03129996359348297, 0.009799017570912838, 0.3463650345802307, 0.12315808236598969], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f00afa54b823a3e2db7ba9e9d869226b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.447824090719223, 0.00880325585603714, 0.41575488448143005, 0.2415114790201187, 0.041501741856336594, 0.1451655626296997, 0.383537620306015, 0.17989987134933472, 0.32615160942077637, 0.03245794028043747, 0.4895342290401459, 0.29344645142555237, 0.05454225838184357, 0.12427417933940887, 0.40294864773750305, 0.4035537540912628, 0.10537374764680862, 0.393800288438797, 0.3693407475948334, 0.4245782196521759, 0.3908657729625702, 0.16355445981025696, 0.1863519549369812, 0.08279929310083389, 0.1167353093624115, 0.2935860753059387, 0.26003605127334595], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_85e8663c4b36aacb75e92536f9bdb660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4e79002413a331a1e7bed93eb7a6df9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32183587551116943, 0.07812762260437012, 0.35579022765159607, 0.40343397855758667, 0.3156796991825104, 0.49226436018943787, 0.15802668035030365, 0.2576165199279785, 0.46401533484458923, 0.15523691475391388, 0.3138130009174347, 0.029038680717349052, 0.3961853086948395, 0.3033497631549835, 0.1286131888628006, 0.09684892743825912, 0.041613075882196426, 0.07630462944507599, 0.38042861223220825, 0.03967138007283211, 0.3631485402584076, 0.18851487338542938, 0.4516206979751587, 0.010238890536129475, 0.2147141844034195, 0.11866331100463867, 0.46763694286346436, 0.2110256403684616], dtype='float32').reshape([28]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b663889ac0db9e1efb78fa7c48c7e321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 11109, 4, 17], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3100c2a4cb6b0d6e6a91712d2272267a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2117374837398529, 0.3110109567642212, 0.08005603402853012, 0.049314819276332855, 0.17979243397712708, 0.4350840151309967, 0.13236737251281738, 0.4425632357597351, 0.05247174948453903, 0.4794124960899353, 0.19003576040267944, 0.07604288309812546, 0.02537468820810318, 0.09547857195138931, 0.4301227927207947, 0.22933462262153625, 0.20676052570343018, 0.16440322995185852, 0.44798508286476135, 0.4390574097633362, 0.3985956907272339, 0.4156531095504761, 0.1528995931148529, 0.2105870395898819], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b5f63cbadb0d36809db2d4093236555a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 8, 7, 8, 7, 96], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_91e6eff5aee7fe8e518e8dddae5b7bcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 64, 49, 3, 3, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_a56ce3aa80ded6a43d4760fa67f2a2f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 2048], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1280, 32, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09c15212db8e4116d717a2ec81cfd350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2de81fcc00d7ffa4a406ea4659814fd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 7, 7, 768], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_99ee807734b4b73f3aa2306f8e5ebe73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 768, 7, 7], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8dccc8bc5773bc4065087b4e7bb99306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_faab50eb1634d23a21830a8e3c0ddf64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([392], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f1757dfd41de09da4b336b03b7fcbb3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 392, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 8, 49, 28, 28], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_cfc5963bf14aedc371dfe3c03682fcc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 6272, 784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 8, 16, 49, 28, 28], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_3a67e06620b1e4d7c970689b0e0bced7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 128, 28, 28], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_686e383e15a67641022e6325124aaa73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 2, 7, 2, 7, 384], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_31d8bb07d21e80ee441322f58a3742c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 1152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 4, 49, 3, 12, 32], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_550e84987ba6298abb11dfc4bafc6f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.430882066488266, 0.24057109653949738, 0.4036540985107422, 0.26198554039001465, 0.19838570058345795, 0.43231403827667236, 0.12207135558128357, 0.34702810645103455, 0.21440069377422333, 0.10731878876686096], dtype='float32').reshape([10]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c0156cfed6741fd3941c393340f4d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5e0fbf197bb5e5928d039d339d6b2eeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 7, 7, 768], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_77b54dd702491a6168b1213ec4ee7d98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05662224069237709, 0.4162944257259369, 0.08763735741376877, 0.17152777314186096, 0.09010167419910431, 0.4844622313976288, 0.055485934019088745, 0.4896796643733978, 0.3043496608734131, 0.45733919739723206, 0.44492271542549133, 0.3385728895664215, 0.4157136380672455, 0.4677450358867645, 0.15667876601219177], dtype='float32').reshape([15]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_559e42e307f82499bea2af87852cfc7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.005771264433860779, 0.2253718078136444, 0.32170209288597107, 0.06335724890232086], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b558a5fb8d6ff35fd0434e999f6bc94b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27658316493034363], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_99bf2cfca4b8169cada955bee139d814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1568], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 28, 28], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a6cbc94131734beee8b00cff15102a5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c7ce804f9b0f6be3beb30396abf2248
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3756915db2377377d6e495897e11aa57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c7ce804f9b0f6be3beb30396abf2248
    def get_inputs(self):
        return [
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_0e817a5dcc0591cf7fe543154d265bc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c7ce804f9b0f6be3beb30396abf2248
    def get_inputs(self):
        return [
            paddle.to_tensor([384], dtype='int64').reshape([1]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_2de10bc1390fe265154703841e6aa2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([44, 7, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4, 11, 7, 7, 384], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_f8f40ce373f85f55882a182ed663d254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c7ce804f9b0f6be3beb30396abf2248
    def get_inputs(self):
        return [
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_c403b7478fd3a98885460673d8fef80c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c7ce804f9b0f6be3beb30396abf2248
    def get_inputs(self):
        return [
            paddle.to_tensor([77], dtype='int64').reshape([1]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_305cbcd115e856a2e5259572f49973a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 11, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 28, 77, 384], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8ac1266716ee79e2ff481ee88f188141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 4, -1, 50, 76], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_143dc2abfdf4ccbadc22ebc03167922f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 20, 50, 76], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b8b5ee5da544ef47f7aacba849872f4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13316190242767334], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_63b8b5a5027f0cbb5fc7ecc55842c569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 49, 24, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_251afd04f8bc95c955f8aa07b5f7615d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1536], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 49, 2, 24, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_02cd140441f0449b57fd0e83c4c04f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4799157977104187, 0.4207037389278412, 0.28061923384666443], dtype='float32').reshape([3]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_85e8663c4b36aacb75e92536f9bdb660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e40e301a97618fe012b88ab084ec1d13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([100, 152, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7cd9dab09dcf10496e93f70721b19091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([100, 152, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cf5c60a007a5fd5705708385874cb5c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([50, 76, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c87443c8b7ccfcde19210547a87c96d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([50, 76, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e6616d585bf1c52ac02f9c52a301cb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([25, 38, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4fbfb5c9bec96f46ce97b0f1dd95c634(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([25, 38, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5d9a4774c229d2426f9d787e857aaffb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([13, 19, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ac9ba2e8ceb45b0ab7e7cf86f484a25d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([13, 19, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c3a9314965492e4eea319c340153205d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([7, 10, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ab2c9991bc14e72db1baabaefa8b6d82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([7, 10, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_02cfadd927e4ab82bf663ebb392d603b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9b97eb0c2cb43a89c4bc7f14b85a6a47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 784, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 16, 49, 14, 14], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_4a2becc6197d206875132adc55030386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 12544, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 16, 16, 49, 14, 14], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_107f70e055e74c695fc7706f8eee0d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 256, 14, 14], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b603c0fd70d78e6a3f9de386171905ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cbedb7cba785e938c357ca14b4c91c3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4580027163028717, 0.16451968252658844, 0.42254188656806946, 0.465837299823761, 0.07051167637109756, 0.07370806485414505, 0.21749553084373474, 0.35070130228996277, 0.18168942630290985, 0.007262229919433594, 0.1682676374912262, 0.07652907818555832, 0.13684941828250885, 0.3156600892543793, 0.3529122769832611, 0.07527747750282288, 0.05394739285111427, 0.19314171373844147], dtype='float32').reshape([18]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1d35570dadea980ad61998cdd7c6f5d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_39c6d15321088eec074af2f7e62a775e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1568], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_78e142bc3ad1c821e5123eee45f7f318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 1568, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 32, 49, 7, 7], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_aa283396a77a372d81b303b6922ac96c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 25088, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 32, 16, 49, 7, 7], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_141c1f0859b682cf7a65555c133dee49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 512, 7, 7], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a78577095dd73ab704e27bed772eedcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1265576332807541, 0.25555220246315, 0.4160109758377075, 0.311381071805954, 0.11825262010097504, 0.021214604377746582, 0.29918426275253296, 0.3300848603248596, 0.1980704516172409, 0.1885392963886261, 0.030201060697436333, 0.035927630960941315, 0.35452866554260254, 0.4204668700695038, 0.1690972000360489, 0.27437615394592285, 0.1120743677020073, 0.02278437651693821, 0.40014153718948364, 0.028624573722481728, 0.24875149130821228], dtype='float32').reshape([21]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1623e4761fe744cf7b01beaac059c59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([144, 96, 4, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 96, 96], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_717a27c828ce9ed6bfdba8be5d874356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([144, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([6, 1, 24, 48, 2, 96], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_aea141952e703ac656b02ec4b7a996a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb95bdcdf1ba05c978d79df93030e3c
    def get_inputs(self):
        return [
            paddle.uniform([6, 1, 48, 24, 2, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([6, 48, 48, 96], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f311db7c73898b572af69b4603d1ea93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_fdcd6150f0bdb6fffc8be48ce8c6e3fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 4096, 5, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7f5425d516374931f2038b27ac637dfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 4096], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 320, 32, 128], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b4d48f961bdd6507ed527888faea93b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_524ef0a2171baa2bf93ddd14fc3ae44b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 320, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b0ce22f5ea619cf6e2222d9d9d8b7c0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 640], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 5, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_4245f6f9320df33d4ef0d93d0b09fba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 5, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 4096, 320], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_99f435e2e4325f9bbdb2673ff77027e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.38644328713417053, 0.2955161929130554, 0.35687878727912903, 0.09353286772966385, 0.012353244237601757, 0.21069584786891937, 0.13638833165168762, 0.25171399116516113, 0.450852632522583, 0.32727208733558655, 0.4782477021217346, 0.4848768413066864, 0.061671119183301926, 0.2639332413673401, 0.03905918076634407, 0.2495390623807907, 0.41244444251060486, 0.42848628759384155, 0.43740710616111755, 0.31789618730545044, 0.24425235390663147, 0.23804932832717896, 0.15056902170181274, 0.25151747465133667, 0.10227960348129272, 0.49382084608078003, 0.22057569026947021], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f7f62f2f0f020f9e6a466633eb6ff475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_15c25f752594f1b96beb4dd3fc6b16db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([163200, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0f9af46899bd34fe1df62bad46f237de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([40800, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d82df8600e1a88c4f14ee178c76265ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([10200, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c443d5e95045afb10f3f1c4ac20e3c4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([2550, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8f29386b04dc40530a2c0758071f4594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([663, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ed1f3817077876cb2573fbb30210d086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 272, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a0c0d1aa9725c33f72807bb0888d7f65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 136, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_61d0c2b47444cd8d4d5c90249f5d7b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 68, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_87ad536011c78fe555bfd9905c0984e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 34, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_ca9b0071282e13530d402a3cc0c3f268(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 13, 17, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_13bbc857b74101c878940338f68d711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 272, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_dcddfe4f4ba4c7491084ac43fd84d9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 136, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_1531ef36b7a8bf0045b3423811548732(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 68, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f72f81564cd2db7f0844cc18d9c1283c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 34, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_95871f614bf2178dafe31328b59cab5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 13, 17, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_fab30a2cc990466f95b2f07ffc05ed3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 4096, 5, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a2d18946848c5aa7139bb3774ad5d69e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 4096], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 160, 32, 128], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f80261241dca370c98381bc30b86e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d637bac5e5314f2abe57bc548c6dab44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 160, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f2ce43c2835e7db434475de83cfe07b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 5, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_fe826a55d0c7a0eef9d277856415640c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 5, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 4096, 160], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f311db7c73898b572af69b4603d1ea93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a4fe737c29f2de0610de8192d9e1d805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d402804a7e63d82310a2fc3a4ca6c154(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.36889195442199707, 0.32314151525497437, 0.40343788266181946, 0.05323147028684616, 0.44890162348747253, 0.4594721794128418, 0.18944327533245087, 0.21459008753299713, 0.4639977216720581, 0.38743171095848083, 0.30398663878440857, 0.24411188066005707, 0.07281976193189621, 0.12250758707523346, 0.14254792034626007, 0.2048199325799942, 0.2864634692668915, 0.11368405818939209, 0.16232971847057343, 0.2848687767982483, 0.25139281153678894, 0.42650091648101807, 0.19107483327388763, 0.13322541117668152, 0.4958347976207733, 0.07153818011283875, 0.2933686673641205, 0.4006480574607849, 0.3107614815235138, 0.17132686078548431], dtype='float32').reshape([30]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_570544b38ffa0b8d495e55315da58c5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.36460018157958984, 0.24206413328647614, 0.15391521155834198, 0.31016355752944946, 0.4928230941295624], dtype='float32').reshape([5]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bcd563fc7450ef5204b1bf3afb7988c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12124917656183243, 0.31849154829978943, 0.006244766525924206, 0.43679264187812805, 0.31565794348716736, 0.16388705372810364, 0.048823412507772446, 0.08508273959159851, 0.39690691232681274, 0.24773459136486053, 0.02476266585290432, 0.2776627540588379, 0.1688499003648758, 0.005827212240546942, 0.4222530424594879, 0.0265923123806715, 0.2477402538061142, 0.31818878650665283, 0.39091768860816956, 0.18599413335323334], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_13ea57928908ac035f76f2f5e89ce357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0014678509905934334, 0.16182254254817963, 0.28805601596832275, 0.05834173411130905, 0.4069727659225464, 0.1331353783607483, 0.06183205917477608, 0.10767656564712524, 0.29625195264816284, 0.1874147355556488], dtype='float32').reshape([10]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c0156cfed6741fd3941c393340f4d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b574173db4e4c22e174d94b991591891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25499093532562256, 0.3293091058731079, 0.24197892844676971, 0.26669764518737793, 0.039081428200006485, 0.4896862506866455, 0.3725642263889313, 0.2030593454837799, 0.45584502816200256, 0.4110562205314636, 0.010091304779052734, 0.17832723259925842, 0.46857357025146484, 0.20732809603214264, 0.27470719814300537, 0.18708191812038422, 0.33648279309272766, 0.07515405863523483, 0.0906173586845398, 0.05673728138208389], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_db1388a65a19770df872c92d63ea55d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 20, 128, 256], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_ca82a4da28970b48bd75e52ae8bfebe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 2, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 40, 128, 256], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1e2c7f966d81db07ffdd7e160cd0181b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 40, 64, 128], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_8c2e63af0d0bee98a4f0c30d350a56bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 2, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 80, 64, 128], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_430b978457976f24aac3b5bf75f4b1ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 80, 32, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_8831deb64d6d534db0e3b71b7a6c85f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 160, 32, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8fb7a28374a53e270a0efcec85e87991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2530386447906494, 0.25541046261787415, 0.32833072543144226, 0.36849626898765564], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4ee3b807c30c71482b85c15453249e3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f10b91eeaa45ba5106bd12c92acf2d26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_940b65524433657086eff3256fa78c3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_c0156cfed6741fd3941c393340f4d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f80261241dca370c98381bc30b86e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7fa51902d472f1640d95a3b92b92da33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17080792784690857, 0.22578401863574982, 0.1291782110929489, 0.4343526363372803, 0.2868131697177887, 0.28507000207901, 0.30904752016067505, 0.33205220103263855, 0.10103096067905426, 0.3606058657169342, 0.462294340133667, 0.23810136318206787, 0.2268831729888916, 0.3057282865047455, 0.07477088272571564, 0.25593119859695435, 0.005254105664789677, 0.33009669184684753, 0.17935186624526978, 0.15293264389038086, 0.25523579120635986, 0.4296950399875641, 0.46517860889434814, 0.2439187467098236, 0.4688497483730316, 0.007156228646636009, 0.4462352693080902, 0.40993165969848633], dtype='float32').reshape([28]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_dde5464c9b5ac186e34b008081cd2c7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01637609861791134, 0.001833584625273943, 0.2879737317562103, 0.4342327415943146, 0.38258132338523865, 0.47407469153404236], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6772bff0b82ca1d3b76a7d419f5b974d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1137811690568924, 0.19190220534801483, 0.49950331449508667, 0.2924526631832123, 0.2634580433368683, 0.3164956569671631, 0.1757170557975769, 0.04516018182039261, 0.07830245047807693, 0.2540324330329895, 0.3775123357772827, 0.014712564647197723, 0.26110321283340454, 0.2782862186431885, 0.19824713468551636, 0.2879493534564972], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_dbec0c5ad84ff6f21f03903ca8d46775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 196, 12, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f561dfce2dc5f61884817f6dedcfa452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 384, 14, 14], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a94745db582e62919bf52edec9b9ce96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 384, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_1c009caf4435d3ba6206f215d613303e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 49, 2, 12, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_e6a5f950b2dfc7ef3d7318353c1872ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 8, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2436d74db4e929ac817aecd5742bb88d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 8, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_4ea02869efdaae438427608291139af8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 512, 512], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_faab50eb1634d23a21830a8e3c0ddf64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([392], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f1757dfd41de09da4b336b03b7fcbb3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 392, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 8, 49, 28, 28], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_cfc5963bf14aedc371dfe3c03682fcc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 6272, 784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 8, 16, 49, 28, 28], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_3a67e06620b1e4d7c970689b0e0bced7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 128, 28, 28], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_375ec8491881325c77750547bbaa3a34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03578389808535576, 0.34758785367012024, 0.22608201205730438, 0.4792290925979614, 0.40371599793434143, 0.12113098055124283, 0.41606760025024414, 0.16840282082557678, 0.05776949226856232, 0.06135179102420807, 0.26364758610725403, 0.002239569090306759, 0.015230737626552582, 0.37164056301116943, 0.35035473108291626, 0.29125794768333435, 0.46444785594940186, 0.14988641440868378, 0.49450308084487915, 0.21132108569145203, 0.14638252556324005, 0.24559570848941803, 0.4602144658565521, 0.12777304649353027, 0.24181008338928223, 0.4711342751979828, 0.382273405790329], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d8c3ad013e5e88f8095e74d509cdf926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_37dd8ade427fac57583faed4890971dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_17d2c3a610934bc036c4d81d05b8786c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([92928, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_67b828e0eb32446352be5c21bfd27b42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([23232, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5123d3c9e7226d93707449a358af6ac3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([5808, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_22a426f3d731b012265ac8698d7f9691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([1452, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fe3ad97b0035760ff8d5d76a1eb779fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([363, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d2a980d7cec6b7a0d63ebbedf73c3ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 176, 176, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5b43e3530e6d3e131590e1251f683ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 88, 88, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_86d85173a80f93e23cf5f7e7d193786a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 44, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_07761d5d702b74f55270811bb8d5283f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 22, 22, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3320d6f2324dd9b492ce204930f1b44c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 11, 11, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5c79105794c661914d6ce89ca6b57ffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 176, 176, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_753d8894ad81be4c0408f7fbde743cb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 88, 88, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_72e0b82a9d0c3b244af7b3cf6ca3858a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 44, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e37a4f17c61efc31f7cb78f141f4e8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 22, 22, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7e43b37c35f9f7182cf19109055cae3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 11, 11, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b704e8a2057ee823f83fcb49cedfa20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_37dd8ade427fac57583faed4890971dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8f4b87f8cfe02680f4925ec39785ca39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d07db4fbfad8a4199c3c232094ee1080(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a3ab1719b3f7d0fe3bf959e53457e952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 28, 28, 192], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c6ec9f716c78e418e32e7c85f7037e07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.012862122617661953, 0.3486284911632538, 0.1480158567428589, 0.3246766924858093, 0.24588543176651, 0.13814307749271393, 0.30698448419570923, 0.008717376738786697, 0.09988720715045929, 0.047545261681079865, 0.09192150831222534, 0.42906248569488525, 0.24343986809253693, 0.12275784462690353], dtype='float32').reshape([14]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a63c57602a1162b7aa44a4a5a3d89db0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_23e407b2600c8c5e2eb09a2ac7c0ac73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a4fe737c29f2de0610de8192d9e1d805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_87ac56114947f955f16db848380a15ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 784, 6, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_582c4d804520c04931648161479d67ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 192, 28, 28], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_69ae06f9483c5274260de6bbe41421ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 192, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2d9a363e54c4eb4a331f0c0fafb73e66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 49, 2, 6, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_82519c241c909029133986ca4c99dc8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4014669954776764, 0.07139602303504944, 0.29676905274391174, 0.4604503810405731, 0.44008669257164, 0.2944583594799042, 0.24871909618377686, 0.17110446095466614, 0.10190865397453308, 0.11969157308340073, 0.3217734694480896, 0.08458134531974792, 0.21591755747795105, 0.2321287989616394, 0.4024973213672638, 0.32377389073371887, 0.2638707458972931, 0.37797221541404724, 0.31608474254608154, 0.022159092128276825], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7965c26b3fe827dfa84d9a5c7e72ebfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 232, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 116, 32, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a0833dd5f455e98f83c325dff7ba02e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 116, 2, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 232, 32, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ba117c9ce7315fb172057dd48c5e91e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 196, 12, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8676fc8188c783c45ad6d1a6df9cf621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 384, 14, 14], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_194b6323d861b9a7604250a0d05b7ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_76fef51d3105613aa1e0831b91f8fbf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 384, 49], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f2c6be669417fed3e006e1064610b2c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 49, 2, 12, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_11e3cead3c782fd54fe49af070fbfc12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21500086784362793, 0.04354821890592575, 0.10402407497167587, 0.40182361006736755, 0.34149643778800964, 0.2719837427139282, 0.049481648951768875, 0.4836325943470001, 0.12802501022815704, 0.19191297888755798, 0.2708819508552551, 0.3976342976093292, 0.03494085744023323, 0.33076658844947815, 0.35550862550735474, 0.3499668836593628, 0.015214234590530396, 0.12706783413887024, 0.3204815089702606, 0.06064462661743164], dtype='float32').reshape([20]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b4d48f961bdd6507ed527888faea93b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_945fca59373e41f913991129f8a1c012(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 56, 56, 96], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3327aae5c6d581147e4dd49f67abb766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4485843777656555, 0.0359734445810318, 0.31495726108551025, 0.487763375043869, 0.36191126704216003, 0.47953173518180847, 0.041182149201631546, 0.15631158649921417, 0.05624333769083023, 0.3732539713382721, 0.15447430312633514, 0.3884110152721405, 0.07563556730747223, 0.0035383461508899927, 0.36508822441101074, 0.48451176285743713, 0.2658371329307556, 0.26387733221054077, 0.4860225319862366, 0.4527783989906311, 0.2296392023563385, 0.28179851174354553, 0.32420122623443604, 0.3536281883716583, 0.39163273572921753, 0.36755645275115967, 0.008858303539454937], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d8c3ad013e5e88f8095e74d509cdf926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_23e407b2600c8c5e2eb09a2ac7c0ac73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5617cb6a533ae50d3ed418a5373a71cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10533525794744492, 0.37207502126693726, 0.37158074975013733, 0.19177091121673584, 0.17983676493167877, 0.31317439675331116, 0.4399746358394623, 0.47251495718955994, 0.13902905583381653, 0.13061034679412842, 0.4526698589324951, 0.3094407618045807, 0.29889750480651855, 0.0033842038828879595, 0.405357301235199, 0.22110812366008759, 0.10751049220561981, 0.04402869567275047, 0.34031006693840027, 0.2686481177806854, 0.49320927262306213, 0.4958127439022064, 0.34676581621170044, 0.12533695995807648, 0.29797276854515076, 0.48529496788978577, 0.3529146909713745], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d8c3ad013e5e88f8095e74d509cdf926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_82ba62083567979a7a351ff58220ab8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4010365903377533, 0.03905879333615303, 0.42070919275283813, 0.05408880114555359, 0.27534979581832886, 0.4043888747692108, 0.27542543411254883, 0.40311601758003235, 0.12169386446475983, 0.01518539059907198, 0.23841431736946106, 0.052729785442352295, 0.37454599142074585, 0.3134559690952301, 0.2207188606262207, 0.20049595832824707, 0.2805255055427551, 0.463447630405426, 0.14539524912834167, 0.36933547258377075, 0.44843557476997375, 0.4517950713634491, 0.11792438477277756, 0.3528826832771301, 0.14129431545734406, 0.4044919013977051, 0.3259495198726654, 0.017275767400860786, 0.3249645531177521, 0.3696381747722626], dtype='float32').reshape([30]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e5ec1e605f87a274c28649cf9bc487d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ab89296a48027a3c8439ae8d28d21342(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 192, 28, 28], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c0156cfed6741fd3941c393340f4d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f80261241dca370c98381bc30b86e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0fa1a9074d56a30ddfb441a4bd37d368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([8348], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0fa1a9074d56a30ddfb441a4bd37d368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([8348], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7446fcacddf0ef57f2477379629c950c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([141916], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4, 17], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_0fa1a9074d56a30ddfb441a4bd37d368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([8348], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_37dd8ade427fac57583faed4890971dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f8bbaa7d152a086cb15c52d1be282c47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 16384, 2, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2666730c5d87c8eceb5b8d9d11fd8d0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 16384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 64, 64, 256], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b1fd6939043597cb63b15f0b361b19e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 64, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_0075ad90c07fb9f78cffd87daaf9dd14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 2, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_c012d0442b9df03b45403e428dfde57e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 2, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 16384, 64], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5116cc3b00d34e35e16e9ecfffa2ec44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 8192, 2, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_92d54773f3c8f01bd93bb8106881b180(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 64, 32, 256], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f6d10c618bfaff6e08ed38befb215f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 64, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_de2925b336f80a93efd7b2367e9c1699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 2, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_060bcbb131d4bd44da518758c542a5c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 2, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 8192, 64], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e5ec1e605f87a274c28649cf9bc487d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_818e77366e45ddebdf3ec562e9efe7c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([154560, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_67bd00364eb79f5304f5f082ec371900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([38640, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_92bbbedcbacffbc62902a862c8f36280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([9660, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_48a2d465128b04c5742fa282a3dceab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([2415, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_67d3d57f26ebea55d6c35cfeb3455720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([648, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_adfd0b62b3f14a1ce9c36cc573312ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 184, 280, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b4301dedaa6c715fd5ba8cc39b2c42ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 140, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_90e584c9f6b9d1a82ec3d24de9944ed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 46, 70, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_676c9f4fdffc401a9d48f25efb0084b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 23, 35, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_4bcbe3605b28a41732c51413f7323451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 18, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_d063259157dcd084cfb2b6b03ff19cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 184, 280, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_d570cd2b264dfc02e535e56a987309e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 140, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_dfb303ce64a308529c57e352678a2b84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 46, 70, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_4bdf9d5cbbcf05a0efb97af4ecf808d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 23, 35, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_985e74b0615f6320a7192612325fcceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 18, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_71c4926e0214d12c84409b01a4fff2fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30699923634529114, 0.48432669043540955, 0.27645936608314514, 0.24580760300159454, 0.003586416831240058, 0.260032057762146, 0.4442516267299652, 0.31632092595100403, 0.43949776887893677, 0.3274436593055725, 0.3775838613510132, 0.20359082520008087, 0.48222967982292175, 0.22325760126113892, 0.09545280784368515, 0.36243534088134766, 0.060409195721149445, 0.1802220642566681, 0.010436407290399075], dtype='float32').reshape([19]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_78b037098246332bf8bcf356401028c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03301683068275452, 0.15708240866661072, 0.11235228180885315, 0.1250280886888504, 0.34081128239631653, 0.4757131040096283, 0.29694339632987976, 0.2446458339691162, 0.23785564303398132, 0.12030619382858276, 0.052959196269512177, 0.342658132314682, 0.37153875827789307, 0.016528453677892685, 0.39501458406448364, 0.44962242245674133], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c9fdb4a27124c40e1ced851a93aa1588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22980962693691254, 0.1062212884426117, 0.3155204653739929, 0.14105702936649323, 0.23150524497032166, 0.3753844201564789, 0.4529428482055664, 0.13754242658615112, 0.06424175202846527, 0.48614591360092163, 0.3829892873764038, 0.3660324215888977, 0.1641504317522049, 0.26202064752578735, 0.3046455681324005, 0.018520545214414597], dtype='float32').reshape([16]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0b1acb06b0664b5c49b3197f81a13f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3a35335067055f5cb449efe4dff75dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3d004be3460d36174e4bc9ff38eefe8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 16, 64, 320], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6ee4a93e6d23e9398b329c615288be9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 144, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 128], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_64831f000878442c6f26321f5339d297(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([43776], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1b26f838b88dde21e967f5b20306ecb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 197, 3, 3, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_e6b6cd365b345d9f54ed9158f1b4faa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 3, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 197, 192], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_721bad1dbd0e416bcad593bef61d12c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 32768, 1, 32], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4a0dcecc536759577ecac593a4826e10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 32, 128, 256], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e93e14d1d11ea2f83a3af314a07c2fac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 32, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e2a1648adb2ebe72736676c33d3f88ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 1, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_8f1b2866aee7c733a4777c32a4cccbdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 32768, 32], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b235860d14a565ae3551e71be4480c65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 512, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7df3ccc1ddeb8b527b58783fcb59faeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 512, 64, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e8ab3aaa2ad945e3597d4628aadfb89e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1565646529197693, 0.0027627793606370687, 0.3374702036380768, 0.20296543836593628, 0.3980843126773834, 0.17129798233509064, 0.09815063327550888, 0.018883714452385902, 0.27935990691185, 0.24570538103580475, 0.4974592328071594, 0.10620273649692535, 0.48902755975723267, 0.3203522861003876, 0.03693389520049095, 0.1185489147901535, 0.2016516923904419, 0.1601341962814331], dtype='float32').reshape([18]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_499abd9cd45c35af4edd9b04512c5a3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 18, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 2, 9, 112, 112], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_5ee4dd6d9707bd8b18b3f1901240525f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 288, 12544], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 2, 16, 9, 112, 112], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_1c348c426ebb8f0dd68144b450535f61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 16, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 32, 112, 112], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6491e511097d8687bd23ddc7c06c6d66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 8, 64, 320], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eff5d5ed1950f91e9bf10b328938b803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f5407c3c91fe58d6392ae160eb6f744f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0634734034538269, 0.10003334283828735], dtype='float32').reshape([2]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_066235bb6d7f2f6eb76672a92ba0458f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08166893571615219, 0.43015405535697937], dtype='float32').reshape([2]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5a0a19822afaff3e1cb27eca731583c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08155101537704468], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_de66ffaf37abcc10db3c9d29fcf9e278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29428690671920776, 0.41796788573265076, 0.20266442000865936, 0.2329687774181366, 0.3596192002296448, 0.4246879816055298, 0.4193873107433319, 0.15613143146038055, 0.19018203020095825, 0.18025170266628265, 0.008102930150926113, 0.35694602131843567, 0.3762901723384857, 0.42119014263153076, 0.027734167873859406, 0.45182931423187256, 0.013628660701215267, 0.2835884690284729, 0.08778959512710571, 0.21491748094558716, 0.38658884167671204, 0.4129372239112854, 0.3228364586830139, 0.2303777039051056, 0.2857441008090973, 0.11823705583810806, 0.10454075783491135], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_85e8663c4b36aacb75e92536f9bdb660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2d025537a0b06059c49c140194ad4421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d08255f9de23ebb3f061ef4a588e620(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_524727cd00f23fb6eab95854ee357979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 14, 14, 384], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5f82c7ed23915bf3393280694edd7e31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 3024, 4, 17], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cb0de25b3fcc78946f96c4ee700819d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 196, 4, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_67dbcb4c26b6abfea3709e73bfb7a54e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([4, 38416], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 196, 196], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_67885294bc04c397eaa071255384cd0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_5bc320b26e1dd96923a1d619e75f258f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([400], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9c3661ad98ccf19b17c4a35675d2f11c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([165888, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_043fa97cc586bded3aceade5aed48bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([41472, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fb543e76c2ea295c5fdd4b8b7d86895d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([10368, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ae4a4e486f07389d87c6e21d9fb84dea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([2592, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_67d3d57f26ebea55d6c35cfeb3455720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([648, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a651c4ffdb1eb31d48f85bb3aec88080(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 288, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_1b84abec72a26e196aa05f709a9ba0b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 144, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_34af31ca2892c1870466c193edd85221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 72, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_e69a1ef827da652783ed65e77267e36c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 36, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_4bcbe3605b28a41732c51413f7323451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 18, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_14f78d09a14cd0bcf8348315c9298b03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 288, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_890247ec86c9741a64b137ec5ef80fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 144, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5011ee5a5982b70907ebb8e39c37257e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 72, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2367a402e9b7932dcdb6b568c94a8179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 36, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_985e74b0615f6320a7192612325fcceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 18, 12], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_45df3ffa15871f14420e05d293f1b5a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([72, 72, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bf4e8d20abd7b3e624a52522fe951ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([36, 36, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_11c5fb6eb1cc326f080eacb00b19fec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([18, 18, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_45df3ffa15871f14420e05d293f1b5a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([72, 72, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bf4e8d20abd7b3e624a52522fe951ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([36, 36, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_11c5fb6eb1cc326f080eacb00b19fec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([18, 18, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7de33014cf2c6b5896a2063f8abafc31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1739046275615692, 0.44292253255844116, 0.31275105476379395, 0.24400556087493896, 0.19731159508228302, 6.67705899104476e-06, 0.48903316259384155, 0.3547435998916626], dtype='float32').reshape([8]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0f9bfed8f4dee95fcff388b4b4d3bc56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28568196296691895, 0.3302013576030731, 0.027214819565415382, 0.4138332009315491, 0.042869821190834045, 0.10298629105091095, 0.3594546914100647, 0.16436240077018738, 0.3079589605331421, 0.2675006091594696, 0.3417028784751892, 0.01723453216254711, 0.18177537620067596, 0.49302488565444946, 0.4460805654525757, 0.042860254645347595, 0.12925837934017181, 0.3306274712085724, 0.4582592248916626, 0.1777510941028595, 0.0368225984275341, 0.3415205478668213, 0.20445339381694794, 0.17541377246379852], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3504c616dc69f34e1044c794c0aca619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 185658, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_48f704e8578991f4e7d01d46fd67c8cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 185658, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b348bd80a035b9e12bb6288cbb5d4bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 3, 8, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_0be8989e71c426cbf242833693f50574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 8, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 256], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b5d3de08abd8810053a095150780a6a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 49, 8, 16], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ebcf1318b4d80b8ae6d04f8fcece597d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([8, 9604], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 49, 196], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_a1b114b7f2e89b4ef750f16c088eea4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 16, 12, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_dad8c140524e765a05fbf92c4088b5a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf0dc6ed7282ec3bc65374d08d0352e
    def get_inputs(self):
        return [
            paddle.uniform([12, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 16, 16], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_39c6d15321088eec074af2f7e62a775e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1568], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1c8561c110dec7fd4d0e9d1fd5b28924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 1568, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 32, 49, 7, 7], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_404abdb05bc7e174198a046c369019f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 25088, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 32, 16, 49, 7, 7], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_56554c8d9fe96c18d78f9e9d9495cc42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 512, 7, 7], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b6bcb0089fba35f1f3cd698f8d2e3e2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4111572802066803, 0.10029443353414536, 0.30790531635284424, 0.3856617510318756, 0.2341999113559723, 0.06075889989733696, 0.429999440908432, 0.13796871900558472, 0.43759387731552124, 0.4527134895324707, 0.08752412348985672, 0.48634159564971924, 0.2569914162158966, 0.28311750292778015, 0.4197201132774353, 0.02713240310549736, 0.4904816448688507, 0.32734715938568115, 0.4286387264728546, 0.24019598960876465, 0.22035078704357147, 0.41939467191696167, 0.2944999635219574, 0.10249026119709015, 0.46785175800323486], dtype='float32').reshape([25]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_67885294bc04c397eaa071255384cd0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_aaabcdd110dc95f8c883b2a71f54f231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 1174, 3, 6, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_85b33573f0b500c59a66ca55cfc49f8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 6, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 1174, 384], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2be732df023c2a345cad67a406ccf0b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c7ce804f9b0f6be3beb30396abf2248
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_70bbb28c536996080893d031f969acac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06992992013692856, 0.16274656355381012, 0.3649451434612274, 0.08391112089157104], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2f042cc013ad701f5265ddbe1c43f0e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4063752591609955, 0.2525491714477539, 0.24242594838142395, 0.1271849125623703, 0.36384841799736023, 0.14547447860240936, 0.46394190192222595, 0.033270254731178284, 0.10290741175413132, 0.20182476937770844, 0.4455929398536682, 0.15107500553131104], dtype='float32').reshape([12]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1d35570dadea980ad61998cdd7c6f5d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1810ccf0eeee1097e95c77e41669f064(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4960156977176666, 0.20150506496429443, 0.17217020690441132, 0.1481056660413742, 0.07369498163461685, 0.11214149743318558, 0.36837485432624817, 0.09432376176118851], dtype='float32').reshape([8]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c0156cfed6741fd3941c393340f4d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f80261241dca370c98381bc30b86e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_b4d48f961bdd6507ed527888faea93b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d5ac830e4c5f34dfe3063933a1525c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.16217000782489777, 0.4453571140766144, 0.13143274188041687, 0.2129790186882019, 0.3854115605354309, 0.3833422064781189, 0.18423058092594147, 0.1925622969865799, 0.18914632499217987, 0.0819990411400795, 0.16699670255184174, 0.26453325152397156, 0.3431435525417328, 0.20854704082012177, 0.454779714345932, 0.17297565937042236, 0.4001516103744507, 0.3465208411216736, 0.3284960091114044, 0.21079564094543457, 0.37737998366355896], dtype='float32').reshape([21]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1cc9913fa8bd3c1df019817d3e97faf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 4, -1, 25, 38], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_84e831cbfc509a20f8f8e1bbc34c0ade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 20, 25, 38], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ba7875a38d3ee908d33fdd46c7333aad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1105346828699112], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_45353e20aa50d583f4709ff76cb768e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f07d1b0c65778299052b5158cb7d280d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_eb7b4c37f952ed11dcc16a836b175492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f6af6188293044829893cc90a67e2ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42264431715011597, 0.11501152813434601, 0.3608306050300598, 0.16361291706562042, 0.035857800394296646, 0.38487017154693604, 0.277341365814209, 0.38714325428009033, 0.11949989944696426, 0.31948617100715637], dtype='float32').reshape([10]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4caca90988c394c1868b4a421e561045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 16, 32, 160], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a1a49987067332c313aa600b08ca04cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_840bc84ae56b1912873b4fc4a6dbe389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22498555481433868, 0.29613375663757324, 0.15396380424499512, 0.47095802426338196], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ca6d39b8e987f6822bc943412bbf8346(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1280469298362732], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c0156cfed6741fd3941c393340f4d78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4f80261241dca370c98381bc30b86e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_018e09fd11ebaa5aa805e3aa573bd9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0d476eb4fd95f23dd2a3e26c9e954a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_37fd4a9d77c65f37bceb1afd85e1648b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4846818745136261, 0.23585817217826843, 0.254557728767395, 0.35119786858558655, 0.15893448889255524, 0.16664054989814758, 0.16330081224441528, 0.004137658514082432, 0.2912030816078186, 0.4802553951740265], dtype='float32').reshape([10]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_fe42f44a94bff3af893587a9551fbc8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 4, -1, 7, 10], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_71864da2ba0aac0d3561c65b9ecde2f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 20, 7, 10], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_99c8db33d8d42ce52b67f91497fd335f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06777893751859665], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7fe2e5d2e385e324b14c044dd5383367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 116, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 2, 58, 64, 128], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a04f8150525984e07fbfba2e5268ac20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 58, 2, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 116, 64, 128], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e64a9921339cf5bbcd175e7c5d1d8fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4148517847061157, 0.37221869826316833, 0.4874425232410431, 0.29649072885513306, 0.022329851984977722, 0.018986370414495468, 0.12138912826776505, 0.07085155695676804, 0.3140938878059387, 0.3039802610874176, 0.16262207925319672, 0.07761567831039429, 0.13060709834098816, 0.07058137655258179, 0.2868139147758484, 0.4205668270587921, 0.36535564064979553, 0.38821840286254883, 0.18862877786159515, 0.34765341877937317, 0.3513076603412628, 0.2527386248111725, 0.2717827260494232, 0.22784686088562012], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_2416d846d73812649f64fe0952045823(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4682934582233429, 0.011456605978310108, 0.33649954199790955, 0.47898542881011963, 0.22809851169586182, 0.0016958639025688171, 0.4349566698074341, 0.42673081159591675, 0.40266141295433044, 0.09634491056203842, 0.05936230346560478, 0.17022809386253357, 0.455485463142395, 0.3092190623283386, 0.35561221837997437, 0.07121868431568146, 0.48551979660987854, 0.39665791392326355, 0.485460102558136, 0.4335234463214874, 0.050489071756601334, 0.4040681719779968, 0.17221620678901672, 0.18082791566848755], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3fdb87ee1c2ec8dc3d79493ad9824f1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.38536641001701355, 0.4623616933822632, 0.16876591742038727, 0.2525574266910553, 0.26149895787239075, 0.02474403940141201, 0.46118730306625366, 0.11370667815208435, 0.3439372479915619, 0.47348204255104065, 0.3745548129081726, 0.47537916898727417, 0.2473655790090561, 0.10462258756160736, 0.377483606338501, 0.2885785400867462, 0.03178626298904419, 0.37243127822875977, 0.07548795640468597, 0.2363874316215515, 0.2574808895587921, 0.2703237235546112, 0.387234628200531, 0.15766288340091705], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7d5803dd1645626369fb67237b43ffa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1653444916009903, 0.2255038470029831, 0.3941905200481415, 0.36936599016189575, 0.1950184553861618, 0.12728260457515717, 0.20374631881713867, 0.3733196556568146, 0.3231998383998871, 0.2320443093776703, 0.45678991079330444, 0.48950815200805664, 0.41946378350257874, 0.1101560890674591, 0.16559427976608276, 0.25380632281303406, 0.20558278262615204, 0.04992234706878662, 0.08447598665952682, 0.44891855120658875, 0.26131361722946167, 0.4115457236766815, 0.3950589895248413, 0.1848994940519333], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_519fe056b3aacf5f5b9ea459e4c0158a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17494626343250275, 0.36216306686401367, 0.3232136368751526, 0.43000584840774536, 0.18502259254455566, 0.4908885657787323], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_622e1bddeb3cf908619b8cec6cb3f1fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3335897624492645, 0.03182823583483696, 0.3474259376525879, 0.4953592121601105, 0.2950001358985901, 0.06897284090518951, 0.4417458474636078, 0.420430451631546, 0.39246195554733276, 0.10012484341859818, 0.2041030377149582, 0.2258676290512085, 0.10799525678157806, 0.35480013489723206, 0.07811453938484192, 0.3751983642578125, 0.42809051275253296, 0.388996422290802, 0.11931005120277405, 0.3167530596256256, 0.0999925509095192, 0.31627312302589417, 0.34512394666671753, 0.47953689098358154], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_32428e709cdcda485d8b56495c24cf4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20934227108955383, 0.02652132883667946, 0.3402882516384125, 0.14422903954982758, 0.12356464564800262, 0.2396799921989441], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8c72017c07fcf0a5b7c98e1450f76186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05302124470472336, 0.02591085433959961, 0.3757885694503784, 0.3568480610847473, 0.37049600481987, 0.321682870388031, 0.07169646769762039, 0.30701279640197754, 0.021901097148656845, 0.26345381140708923, 0.11741532385349274, 0.28026285767555237, 0.12423190474510193, 0.1937820017337799, 0.4232766330242157, 0.14136800169944763, 0.1926240473985672, 0.3859717845916748, 0.3066234588623047, 0.021561967208981514, 0.43195438385009766, 0.40530723333358765, 0.3323405981063843, 0.18499603867530823], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_3878713c3178352b1242747cbab3d978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3779277503490448, 0.15553362667560577, 0.4419955313205719, 0.32303062081336975, 0.4994341731071472, 0.2351422756910324], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_462555e7ca3984350f1f57f826931194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43636244535446167, 0.09001603722572327, 0.40344423055648804, 0.4261627495288849, 0.07860106229782104, 0.3619501292705536, 0.0827021449804306, 0.35579681396484375, 0.2838437259197235, 0.21985980868339539, 0.2873339354991913, 0.24154025316238403, 0.057367321103811264, 0.011423596180975437, 0.19522923231124878, 0.09822019934654236, 0.3769165277481079, 0.3379030227661133, 0.00046532234409824014, 0.366558700799942, 0.08046781271696091, 0.3581957221031189, 0.47041603922843933, 0.46776750683784485], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_ecd0a2822da9d916f45abc9feea1ad87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2740359902381897, 0.1493968367576599, 0.27455586194992065, 0.49646979570388794, 0.28791648149490356, 0.16906963288784027], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f337efd268e315e4aeef03ee7e4e5c29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.37686535716056824, 0.364767462015152, 0.4691812992095947, 0.43126481771469116, 0.04191863536834717, 0.09619399160146713, 0.18329444527626038, 0.3761677145957947, 0.0347675159573555, 0.44765374064445496, 0.2063162922859192, 0.2650836110115051, 0.22694925963878632, 0.06707358360290527, 0.03854988142848015, 0.023158881813287735, 0.4270826280117035, 0.08227191865444183, 0.1580103635787964, 0.42340904474258423, 0.4839048683643341, 0.48079124093055725, 0.30335235595703125, 0.41177791357040405], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_05b835191af3fd3e0ea533a894aef4c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2315797209739685, 0.04054444655776024, 0.028632165864109993, 0.21265752613544464, 0.17155441641807556, 0.48197758197784424, 0.08563379943370819, 0.4187764823436737, 0.09645210206508636, 0.28938257694244385, 0.25479602813720703, 0.1709919422864914, 0.3712829649448395, 0.09780335426330566, 0.15986888110637665, 0.09450390189886093, 0.13075537979602814, 0.21688705682754517, 0.0030219461768865585, 0.18975187838077545, 0.2924681603908539, 0.025586487725377083, 0.07945812493562698, 0.2683858275413513, 0.004188150633126497, 0.13300234079360962, 0.28002962470054626], dtype='float32').reshape([27]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d8c3ad013e5e88f8095e74d509cdf926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_a860957e97c98663cd33e60437b055a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2725658118724823, 0.11680822819471359, 0.26840171217918396, 0.4876953661441803], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1cb061cf85eb24c812d4b51962a843c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47625210881233215], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc1190d0914ad5c772aea204595f92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e5ec1e605f87a274c28649cf9bc487d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1d35570dadea980ad61998cdd7c6f5d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bea396d3bf8c3da193ed4e1d9d376279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35015299916267395, 0.3572719395160675, 0.34082287549972534, 0.014674042351543903, 0.25240686535835266, 0.12197944521903992], dtype='float32').reshape([6]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_741a69f09bdeb606f52c8aaac3509c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6db0404c7b8a5060db26ebe96656a742(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12939411401748657, 0.25829997658729553, 0.08752501755952835, 0.19330260157585144], dtype='float32').reshape([4]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e69a11c91c6ce8dc67490ff5634f43c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04943191260099411], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_c997f4ff8d2f39a24e3a67f6bf002e6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49289172887802124, 0.2825499176979065, 0.2126092165708542, 0.4577012062072754, 0.11182887107133865, 0.24033594131469727, 0.38607916235923767, 0.3651675879955292, 0.16132789850234985, 0.3675231337547302, 0.39212116599082947, 0.3315278887748718, 0.4065202474594116, 0.26964062452316284, 0.2132101207971573, 0.15668007731437683, 0.17482061684131622, 0.09129349142313004, 0.44572749733924866, 0.1976606845855713, 0.03929867967963219, 0.35385313630104065, 0.13482075929641724, 0.11875192075967789], dtype='float32').reshape([24]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0db6b876854815c1e4bd76e1829b35fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([17084], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0db6b876854815c1e4bd76e1829b35fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([17084], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3c5f8ef093abed93ca332e269f59556b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([290428], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4, 17], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_0db6b876854815c1e4bd76e1829b35fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([17084], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b2107358d7204bae3a9a33e68791c63e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 2304], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 1174, 3, 12, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_9945d8478a6e9aa91b4a4052cc1b7efa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 12, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1, 1174, 768], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_6be0c663f99d26d345b0ef1348c1a96d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 65536, 1, 64], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1461dbf297e234616e80eeaed6c0c7ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 65536], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 64, 128, 512], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1b1fd6939043597cb63b15f0b361b19e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 64, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_c727afbe9ad44f2aa887563aaddc891a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2, 1, 64], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_c2fea8fe6a254aea301985e57ddb686a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 65536, 64], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_4ee3b807c30c71482b85c15453249e3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f10b91eeaa45ba5106bd12c92acf2d26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_940b65524433657086eff3256fa78c3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_598c70e3e278e0c01abe7b6cd7944e4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 4, -1, 100, 152], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_c62e7a032dc818ea902980566ed6ced3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 20, 100, 152], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_bc6bcb9d25bba142c7542acd659c5a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6f6ae5c5d1f1daca042f85cfc98d8be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49417588114738464], dtype='float32').reshape([1]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_131e949067d872334214917a5607282c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 56, 56, 96], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_55cdf5837e3b32cdd5b566347cb98597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_cd664a662fbbea631f3d56c19bbd54c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_287462a0cb3f4b943a0e246810c98aef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 3, 8, 32], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_1b83b006f9e8fc0c0d72d695b8a303a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 8, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, -1, 256], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_1b3940c7d4010e5f5ac23b5ae3780088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_02cfadd927e4ab82bf663ebb392d603b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9b97eb0c2cb43a89c4bc7f14b85a6a47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a82db125ee249a3cb07b70fd73ac0080
    def get_inputs(self):
        return [
            paddle.uniform([22, 784, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 16, 49, 14, 14], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_4a2becc6197d206875132adc55030386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 12544, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 16, 16, 49, 14, 14], dtype='int64').reshape([6]),
        ]


class TestPrimitiveOp_107f70e055e74c695fc7706f8eee0d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711a2ccbfef519241b7d4a1697e9b791
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 256, 14, 14], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f4d53dacf8653c48ec383e7bf661a441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 3136], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([43, 96, 56, 56], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_04cfc56293c0cd495096020a705d254d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_aa29576ace2588a3adff055ca015d61c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2310301512479782, 0.015567287802696228, 0.07995817810297012], dtype='float32').reshape([3]),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_d8c3ad013e5e88f8095e74d509cdf926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f311db7c73898b572af69b4603d1ea93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_037f7f4f84a4bade2cae51f9d63d8566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_09fe298a86a54cff363945127a93fe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_10a70dfa27ac103fba0e7326979a32c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_baddc14d8339179acb91d2e215978669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6fbd165e154c3918830a52b91cca77
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 1280], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([22, 49, 16, -1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0b1acb06b0664b5c49b3197f81a13f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3256feed2bb0cea5fd1736c98672718d
    def get_inputs(self):
        return [
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64').reshape([4]),
        ]




if __name__ == '__main__':
    unittest.main()