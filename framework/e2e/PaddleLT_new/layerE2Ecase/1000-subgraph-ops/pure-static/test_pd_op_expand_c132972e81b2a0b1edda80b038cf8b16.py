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



class PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac46fedf8c9dd7403e84cd816dc95054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 23, 23, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_4281cc7c6c840edf10262b9cde75a7c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 23, 23, 1], dtype='int64').reshape([5]),
        ]


class PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_356551588a29b6cd834944fab6befa66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 60], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b3245e019dc73a829625d4a3e8a68259(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 60], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c238042958f32756b33c63a9283c6079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1542, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_00433c227fec744bd6605bf50e319cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1542, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_c238042958f32756b33c63a9283c6079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1542, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_00433c227fec744bd6605bf50e319cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1542, 1], dtype='int32').reshape([2]),
        ]


class PrimitiveOp_9f94225e3d64183cd49512b33e350fd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0bf229f3d9583d4baaffaafa33b43bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3024, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_1fa327ebcff8a77192429e28c31a0422(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3024, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_883c1681a5d606d0e3f6b27b70d85fbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 672], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fac9a16b6598f7f8cff15acf72f5caba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 672], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5d724e92bd1eb9dde209f7873e3e4b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2361, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_934523672dbe1c70597285c5201aa889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([2361, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_5d724e92bd1eb9dde209f7873e3e4b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2361, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_934523672dbe1c70597285c5201aa889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([2361, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_fb093b421a6410447da51c9168793ddd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4725, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_01d129735086e8ed6982fa947b7ad339(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4725, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b6e2280132408a0f7dad704aa08fd7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_815cea3147e063a86a791e88512727e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f412f1f68b5b98566e86c46f21467d63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 12, 12, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_f3aa89b7ba8049bd5f4bf7055baabf2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 12, 12, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a22e777de1ebd894a0f40cffc90449fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 60], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5d7aab250d8ca90e53a5457724c05b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 60], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2a530876588195ba9322e58751d1a7d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a67bc363805af6fe0198af584d008bdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([4, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_74b0d23caf06c99d12a76f8a8bc397db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
            paddle.to_tensor([6, 1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_4f9acefd065d5f62e263fd08609d8e7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
            paddle.to_tensor([6, 1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_d06758742ab534fc8dae0294c837091d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_45e82ff12ef059b2ae4906f5beb64ab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_40ac53095409212f7cc32c3036680b80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 46, 46, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_1c9a031b86f0ff555b6a8d6e20838ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 46, 46, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_caf494325cb7785e9738e1ada3860292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 76, 76, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_9b40e6f1447139e8cf061e12cd6900b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 76, 76, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_35b3285d64c7670e92cd722154996224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cd57f4fddfb0dd78fa4565a20bcf1d46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1f071b39f9a91d499f28152cb940f67e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 92, 92, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a3d86efc9f343916510ec800e0e32d3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 92, 92, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_6721873326f6529a7eda3a4d1673e011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([6, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a94ed389f3f6aacb14855cd757013f65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([6, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_31bfa23cedf0ecefef1e8b774d594bc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 150, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a1f42efb23003c55092d78a70f86a83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31bfa23cedf0ecefef1e8b774d594bc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 150, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_daec279770bd710d6c3288db3e89f464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3c413ccb493b35739c0d964a59d30642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4657d80d838a94b2d94ce111e32b7ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 60], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_faeb3deddaca00e5489673df486ecfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 60], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_d331541c7bed18d3aaa9f4bb704f460c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_078082851ddba90c38779a3cfb9bfd16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d331541c7bed18d3aaa9f4bb704f460c
    def get_inputs(self):
        return [
            paddle.uniform([100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([100, 152], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_e00b8e8c418f150763e915f5af824ec2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_baf85968c77a204638e6dcb45fe06150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e00b8e8c418f150763e915f5af824ec2
    def get_inputs(self):
        return [
            paddle.uniform([1, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([100, 152], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1c9ed41706f5fafbd7d770de0104ac82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bb4077a66125e5d6f76db6de5cb6044b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c9ed41706f5fafbd7d770de0104ac82
    def get_inputs(self):
        return [
            paddle.uniform([50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([50, 76], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_629e22fa4ace4d000120785dd6107ed1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_763f77730011fdb9d0679cea87aff3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_629e22fa4ace4d000120785dd6107ed1
    def get_inputs(self):
        return [
            paddle.uniform([1, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([50, 76], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ca61144ac414bc18f4fe3787656ba10d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c0733d5ceeecf703bb8232018935fb5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca61144ac414bc18f4fe3787656ba10d
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0], [464.0], [496.0], [528.0], [560.0], [592.0], [624.0], [656.0], [688.0], [720.0], [752.0], [784.0]], dtype='float32').reshape([25, 1]),
            paddle.to_tensor([25, 38], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_5428a73d27468adf43baaf6b22d6d746(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09eaeabe2ec1a91f7934fc3af26a8766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5428a73d27468adf43baaf6b22d6d746
    def get_inputs(self):
        return [
            paddle.uniform([1, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([25, 38], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ecf42a1dd723f2c39888b3bec020296e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e194b611d96d410cd647eda67920e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecf42a1dd723f2c39888b3bec020296e
    def get_inputs(self):
        return [
            paddle.to_tensor([[32.0], [96.0], [160.0], [224.0], [288.0], [352.0], [416.0], [480.0], [544.0], [608.0], [672.0], [736.0], [800.0]], dtype='float32').reshape([13, 1]),
            paddle.to_tensor([13, 19], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_6d9b2cc05a992f8f5657461e89f07138(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17c05c5b7eb8e79df796757b6eeaa20c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d9b2cc05a992f8f5657461e89f07138
    def get_inputs(self):
        return [
            paddle.to_tensor([[32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0, 1056.0, 1120.0, 1184.0]], dtype='float32').reshape([1, 19]),
            paddle.to_tensor([13, 19], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_42cfd1f7c04102075241c08481c408b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dcc1c98b43e2f4f3754a80ac2f85d583(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cfd1f7c04102075241c08481c408b1
    def get_inputs(self):
        return [
            paddle.to_tensor([[64.0], [192.0], [320.0], [448.0], [576.0], [704.0], [832.0]], dtype='float32').reshape([7, 1]),
            paddle.to_tensor([7, 10], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1e1a45aaee3756d701c9f89bccbb4ad7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a22a1ea1ceca2c1b8414c4abd93d54bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e1a45aaee3756d701c9f89bccbb4ad7
    def get_inputs(self):
        return [
            paddle.to_tensor([[64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0, 1088.0, 1216.0]], dtype='float32').reshape([1, 10]),
            paddle.to_tensor([7, 10], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9a76c2ff3c863a74caa70266c3132ac4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 24, 24, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_31ee92fa226dda05d76b5cd3bc60114c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 24, 24, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_b150bf66ee2058dd92403aa24aab89c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 960], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c48d35c8a69d3bec24b5cdc82451de53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 960], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_989d08469724e4838b11e136172db8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 19, 19, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_62b2c0c29dde3398dcf51196c07dca62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 19, 19, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_e818ffc90dd618af0ce8ecc6fbf67753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2053, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_5157bb2b7631ba5c9be9290f809df450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([2053, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_e818ffc90dd618af0ce8ecc6fbf67753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2053, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_5157bb2b7631ba5c9be9290f809df450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([2053, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_8db9841bf7641e8268f119e4d7a1015f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4116, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_32f87e744a726aadad4ac773fd6e78d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4116, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_6892f9e4c6c48db827ff36b86160fb98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 21, 21, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_0c31fcdc568c4f7e20c3eb99c7832a04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 21, 21, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_84d261535c1d346ab2bdde38e6aa0d0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 624], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e7463d382497fbd05dc7757069246bbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 624], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3d8ede9a515be096e7826dfb2ec8f94b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 60], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_02dffb100fe421b634c888c8530c85d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 60], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d02841026368aa0d02d634b3b672b4ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_10081d7320c6d62b2512d0cfdbc39e04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7245af865c8735fc8dd932a31aaa4b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 42, 42, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_5e1fb3969fcebc4e2154a3ec6c77c1c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 42, 42, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_6dcd37a3d9c4adc7a6c3de8755bf5521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e623d12a5a5305f0480be7a48a960782(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2c3420c71451e159d9d68e330ed2f7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0a8e80655b8d1be9730ad4f843a912d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_143cb78713bf926923ea9d5b7e93a803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 84, 84, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_0c214be839f5bb402cf549749d3a7e0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 84, 84, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_d02841026368aa0d02d634b3b672b4ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_10081d7320c6d62b2512d0cfdbc39e04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 336], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_e0aefb9c5c45691dee0ffb0c0c24a329(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7bc87fbb6ad35a81c4a49820ba13e8f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0aefb9c5c45691dee0ffb0c0c24a329
    def get_inputs(self):
        return [
            paddle.uniform([80, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([80, 80], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_2975927fdfaf74ee8895bca8934b19ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6aa32159e5f844db87cf335bf54b0eaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2975927fdfaf74ee8895bca8934b19ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([80, 80], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_24bffa114da82abfe9d6f5d10b2a54a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d90ac7b2848cd0af1e8edafb044c31eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bffa114da82abfe9d6f5d10b2a54a2
    def get_inputs(self):
        return [
            paddle.uniform([40, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([40, 40], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_0e388dabce35b02761bbd5bfb8feea47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8bf08380e906b0cbff350e582106a4a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e388dabce35b02761bbd5bfb8feea47
    def get_inputs(self):
        return [
            paddle.uniform([1, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([40, 40], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_8c666330cbe8ef485927567c5d1cdc5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b01797294f5976efc70a5749f20525d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c666330cbe8ef485927567c5d1cdc5c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [32.0], [64.0], [96.0], [128.0], [160.0], [192.0], [224.0], [256.0], [288.0], [320.0], [352.0], [384.0], [416.0], [448.0], [480.0], [512.0], [544.0], [576.0], [608.0]], dtype='float32').reshape([20, 1]),
            paddle.to_tensor([20, 20], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_7343476fd8001d05e1a67534fc0a695c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5567d9b0516fe49dc3f75c68d56993b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7343476fd8001d05e1a67534fc0a695c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0]], dtype='float32').reshape([1, 20]),
            paddle.to_tensor([20, 20], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7bc87fbb6ad35a81c4a49820ba13e8f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0aefb9c5c45691dee0ffb0c0c24a329
    def get_inputs(self):
        return [
            paddle.uniform([80, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([80, 80], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6aa32159e5f844db87cf335bf54b0eaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2975927fdfaf74ee8895bca8934b19ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([80, 80], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d90ac7b2848cd0af1e8edafb044c31eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bffa114da82abfe9d6f5d10b2a54a2
    def get_inputs(self):
        return [
            paddle.uniform([40, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([40, 40], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8bf08380e906b0cbff350e582106a4a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e388dabce35b02761bbd5bfb8feea47
    def get_inputs(self):
        return [
            paddle.uniform([1, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([40, 40], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7b64f3fc3909434048f9378f5aed4cdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c666330cbe8ef485927567c5d1cdc5c
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0], [464.0], [496.0], [528.0], [560.0], [592.0], [624.0]], dtype='float32').reshape([20, 1]),
            paddle.to_tensor([20, 20], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3957467c97f087f64f2c60037b5aa9d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7343476fd8001d05e1a67534fc0a695c
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0]], dtype='float32').reshape([1, 20]),
            paddle.to_tensor([20, 20], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4c51a3b083be950463bb29d6fdc397d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([5, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cfecf1e8d3c6b686eddfb12fd5f18100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([5, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_38840c6df89a745481606dac397803ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1825, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_1f351d6379fe7545cc7a250beddebedd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1825, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_38840c6df89a745481606dac397803ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1825, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_1f351d6379fe7545cc7a250beddebedd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1825, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_ac53066394b8ec7d4faa34ee3929a7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([-2.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3549, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5add0819a187947ec734746a7ed5bcb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3549, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_01e59846d88efe3eea59d6e9412ad490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 72], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_26813eb0ef9d942fbe847302b3202aaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 72], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_fe7f08897e8c2dd04ddcec1b617950f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49284f64385d4ff4f15dbcc18609c016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe7f08897e8c2dd04ddcec1b617950f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0]], dtype='float32').reshape([14, 1]),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_7b3c7f33d2f98b4e63b8a2567035c72a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a87eecfc3e1ca8db796d8177588b023(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b3c7f33d2f98b4e63b8a2567035c72a
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0]], dtype='float32').reshape([1, 14]),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_8e3085c5c154f523077fe69944ba7730(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61e7b94d588c0c986812a0f35a848583(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e3085c5c154f523077fe69944ba7730
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.0], [24.0], [40.0], [56.0], [72.0], [88.0], [104.0], [120.0], [136.0], [152.0], [168.0], [184.0], [200.0], [216.0], [232.0], [248.0], [264.0], [280.0], [296.0], [312.0], [328.0], [344.0], [360.0], [376.0], [392.0], [408.0], [424.0], [440.0]], dtype='float32').reshape([28, 1]),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_f7071eae07ad94b7ce7196a83b236497(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5278a85de1594306c667534bb8a61585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7071eae07ad94b7ce7196a83b236497
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0, 392.0, 408.0, 424.0, 440.0]], dtype='float32').reshape([1, 28]),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b7ae22309fc9481a7ad9672206dc173a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39fc0ed07f05c021191c48b9ab9ddc72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7ae22309fc9481a7ad9672206dc173a
    def get_inputs(self):
        return [
            paddle.uniform([56, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([56, 56], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_31e376c84f4d4fb1249dd20d61a5c762(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bde493f015ec9c1345742942964e917d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31e376c84f4d4fb1249dd20d61a5c762
    def get_inputs(self):
        return [
            paddle.uniform([1, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([56, 56], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_229f6c8711cd5c32609149eeca560506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3087, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_04e93253cdb3e6b617c532f4b0bddc31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([3087, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_229f6c8711cd5c32609149eeca560506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3087, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_04e93253cdb3e6b617c532f4b0bddc31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([3087, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_4e68706de0e4d9dfdc5acc135a71c339(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 6069, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b46c6f416ac1c0919da86d211de5147c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 6069, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_fbfc8e7fc73fd2e8eef6705f1571e7aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 48, 48, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_1a2227b0a79e2f50ee38c91310c33424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 48, 48, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_b0d97a66db6ad59d0886d51f69d6d43b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f255c1f5d94d110a99088f66ca00150d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4a109093775473f0690e6239b721b2b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_916889b3e5a81ac29fdb44d50a351233(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a59eb41ce593058621cb1df6770ace68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([9, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bbff6353ef4455c449aea0143ed927b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([9, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_74b0d23caf06c99d12a76f8a8bc397db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
            paddle.to_tensor([6, 1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_4f9acefd065d5f62e263fd08609d8e7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
            paddle.to_tensor([6, 1, 1], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_30c716ddc80ddf62fc1bd73dc9fb01d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 150, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7da130a60eb711d7df2c38609992dae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30c716ddc80ddf62fc1bd73dc9fb01d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 150, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_bed2cf1d86c3a96547091471e104c20a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2119, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_564cbd2bbcef2358dd8bc45803fa5cdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([2119, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_bed2cf1d86c3a96547091471e104c20a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2119, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_564cbd2bbcef2358dd8bc45803fa5cdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([2119, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_8db9841bf7641e8268f119e4d7a1015f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4116, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_32f87e744a726aadad4ac773fd6e78d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4116, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_7245af865c8735fc8dd932a31aaa4b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 42, 42, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_5e1fb3969fcebc4e2154a3ec6c77c1c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 42, 42, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_1f071b39f9a91d499f28152cb940f67e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 92, 92, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a3d86efc9f343916510ec800e0e32d3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 92, 92, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_521e5631875a8636a8f898705854a9ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 38, 38, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_77813509ff6b0c33d1fc3e1aa16d3719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 38, 38, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_cd5fd8582b884f617af95d3d023225b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 22, 22, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_11ccf63b43d1dbc8f0d2062f85b853be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 22, 22, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_8345c5a51872fe5098e1d7bd08591506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_964fa42fa938039d2537bf85c3a4d06a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5f53ef896998d9c46eec8879957052d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4f1c2b08f7355883db19d533a0bda2a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 36], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ca331aa5f1d3f3574ff71efb635a13da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b053b91422370dfac770cc66ecc07856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca331aa5f1d3f3574ff71efb635a13da
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1260e1f7c3449d0b549f06437011aa17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2aba05af2b6fc75f7ce4e9dcde2537a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1260e1f7c3449d0b549f06437011aa17
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_2c819a798b53e114137e607f2f35d983(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbf509ac62f96af3feb683473c15a635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c819a798b53e114137e607f2f35d983
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_623261890a7385c161db7b0b6ecd69b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1489b83e20d976b74070b431b6d5f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_623261890a7385c161db7b0b6ecd69b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_859868069b88f8596f1a0a7e5e83001a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[128, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_607fef45e6bbfba88e0726042d959d65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_859868069b88f8596f1a0a7e5e83001a
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_4aa85cd9313e3f8e664abd2fe54781a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02e51cfa063d50ce4fa5997f396d1434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4aa85cd9313e3f8e664abd2fe54781a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_06e001a472a7a36a172be7e1abc4400b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bb0f2152c5f0482b743784b904178a61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e001a472a7a36a172be7e1abc4400b
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.0], [-0.9166666865348816], [-0.8333333134651184], [-0.75], [-0.6666666865348816], [-0.5833333134651184], [-0.5], [-0.4166666567325592], [-0.3333333432674408], [-0.25], [-0.1666666716337204], [-0.0833333358168602], [5.551115123125783e-17], [0.0833333358168602], [0.1666666716337204], [0.25], [0.3333333432674408], [0.4166666567325592], [0.5], [0.5833333134651184], [0.6666666865348816], [0.75], [0.8333333134651184], [0.9166666865348816], [1.0]], dtype='float32').reshape([25, 1]),
            paddle.to_tensor([25, 38], dtype='int32').reshape([2]),
        ]


class PrimitiveOp_187ae21aabb5e819603345b0bc4fe3ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_659f7b87768592e4dfa35503fb2160f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_187ae21aabb5e819603345b0bc4fe3ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([25, 38], dtype='int32').reshape([2]),
        ]


class PrimitiveOp_d062f67f80416cd56d945458e62759ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d3620aa3fc18904195dbe4dc26118cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d062f67f80416cd56d945458e62759ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8d3620aa3fc18904195dbe4dc26118cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d062f67f80416cd56d945458e62759ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
        ]


class PrimitiveOp_36ad6241bcb8ef2bd7cbe71c9b3718e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3006592332043c00328c0534e94e1a74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36ad6241bcb8ef2bd7cbe71c9b3718e3
    def get_inputs(self):
        return [
            paddle.uniform([96, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([96, 96], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_f9b68c628f94982403321d69de79590e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e152031a0582f34a6f42eee121b59bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9b68c628f94982403321d69de79590e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([96, 96], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_4a9d00a7a9dfc38e1b76a2727dea3241(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6055490448583572ffaf0c1b7b10edd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a9d00a7a9dfc38e1b76a2727dea3241
    def get_inputs(self):
        return [
            paddle.uniform([48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([48, 48], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_fe0a772f81c123e863acc164d333cf77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc32404b996e95f0a17fd539913a4337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe0a772f81c123e863acc164d333cf77
    def get_inputs(self):
        return [
            paddle.uniform([1, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([48, 48], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_72c9193326671e924af6f80db707b3c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aec71f38623d45920555486952f9aa70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72c9193326671e924af6f80db707b3c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [32.0], [64.0], [96.0], [128.0], [160.0], [192.0], [224.0], [256.0], [288.0], [320.0], [352.0], [384.0], [416.0], [448.0], [480.0], [512.0], [544.0], [576.0], [608.0], [640.0], [672.0], [704.0], [736.0]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([24, 24], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_78e281f8df0f3999040c1aba884c6e06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e7bb9730e5149dea0969cfaf19b5ddd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78e281f8df0f3999040c1aba884c6e06
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0, 640.0, 672.0, 704.0, 736.0]], dtype='float32').reshape([1, 24]),
            paddle.to_tensor([24, 24], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3006592332043c00328c0534e94e1a74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36ad6241bcb8ef2bd7cbe71c9b3718e3
    def get_inputs(self):
        return [
            paddle.uniform([96, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([96, 96], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e152031a0582f34a6f42eee121b59bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9b68c628f94982403321d69de79590e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([96, 96], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6055490448583572ffaf0c1b7b10edd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a9d00a7a9dfc38e1b76a2727dea3241
    def get_inputs(self):
        return [
            paddle.uniform([48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([48, 48], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cc32404b996e95f0a17fd539913a4337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe0a772f81c123e863acc164d333cf77
    def get_inputs(self):
        return [
            paddle.uniform([1, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([48, 48], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_37a8cfb5c8e6324d875e7f16e3fca450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72c9193326671e924af6f80db707b3c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0], [464.0], [496.0], [528.0], [560.0], [592.0], [624.0], [656.0], [688.0], [720.0], [752.0]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([24, 24], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_94883ba5ce0d3688a403b5a842dc39cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78e281f8df0f3999040c1aba884c6e06
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0]], dtype='float32').reshape([1, 24]),
            paddle.to_tensor([24, 24], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_13c4b2bbf0aae91ad8469d88674de0cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 44, 44, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_06a0bcfecfba40c9e19d98754f8c1fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 44, 44, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_b053b91422370dfac770cc66ecc07856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca331aa5f1d3f3574ff71efb635a13da
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2aba05af2b6fc75f7ce4e9dcde2537a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1260e1f7c3449d0b549f06437011aa17
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bbf509ac62f96af3feb683473c15a635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c819a798b53e114137e607f2f35d983
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d1489b83e20d976b74070b431b6d5f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_623261890a7385c161db7b0b6ecd69b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_607fef45e6bbfba88e0726042d959d65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_859868069b88f8596f1a0a7e5e83001a
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_02e51cfa063d50ce4fa5997f396d1434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4aa85cd9313e3f8e664abd2fe54781a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8095c5b56c7372d8a89568a5c7956f62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 11, 11, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_5393ef5558919bb609e69eea6fd1dc08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 11, 11, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_43fdfbc9f13c72b804744bdbac029505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_86d1117422380fb0f393956a50fce26d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ef157f51d2bef1633197df5e00741955(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42acb789f5f632555873b984158aa026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef157f51d2bef1633197df5e00741955
    def get_inputs(self):
        return [
            paddle.uniform([68, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 68], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ca9ed1d5ce8ef6e9c53b7169ab758013(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd4c6bf5c051834c4a99c7590ac62953(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca9ed1d5ce8ef6e9c53b7169ab758013
    def get_inputs(self):
        return [
            paddle.uniform([1, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 68], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ccff54264c7394421e70d1f6c809cdee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da6a1775895421ba74e81589c0f2d098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccff54264c7394421e70d1f6c809cdee
    def get_inputs(self):
        return [
            paddle.uniform([34, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 34], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b13b708b8c296174893bb892aa14da44(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_475bcd96f99818aac02b442a15008e0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b13b708b8c296174893bb892aa14da44
    def get_inputs(self):
        return [
            paddle.uniform([1, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 34], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_da0825619a524545647f7152e4dd61a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac164a1cd8e8168c8c64008e58e5cd79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da0825619a524545647f7152e4dd61a1
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [32.0], [64.0], [96.0], [128.0], [160.0], [192.0], [224.0], [256.0], [288.0], [320.0], [352.0], [384.0], [416.0], [448.0], [480.0], [512.0]], dtype='float32').reshape([17, 1]),
            paddle.to_tensor([17, 17], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_79b46aafc6c0c8a63578a1866fd29c0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4528f6784237a0394e9bccf7173494fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79b46aafc6c0c8a63578a1866fd29c0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0]], dtype='float32').reshape([1, 17]),
            paddle.to_tensor([17, 17], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_42acb789f5f632555873b984158aa026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef157f51d2bef1633197df5e00741955
    def get_inputs(self):
        return [
            paddle.uniform([68, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 68], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cd4c6bf5c051834c4a99c7590ac62953(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca9ed1d5ce8ef6e9c53b7169ab758013
    def get_inputs(self):
        return [
            paddle.uniform([1, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 68], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_da6a1775895421ba74e81589c0f2d098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccff54264c7394421e70d1f6c809cdee
    def get_inputs(self):
        return [
            paddle.uniform([34, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 34], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_475bcd96f99818aac02b442a15008e0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b13b708b8c296174893bb892aa14da44
    def get_inputs(self):
        return [
            paddle.uniform([1, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 34], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_00bda42edf92a4cbb0464e2b624a47c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da0825619a524545647f7152e4dd61a1
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0], [464.0], [496.0], [528.0]], dtype='float32').reshape([17, 1]),
            paddle.to_tensor([17, 17], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6fb975309c0be6e30660e4687fc07225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79b46aafc6c0c8a63578a1866fd29c0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0]], dtype='float32').reshape([1, 17]),
            paddle.to_tensor([17, 17], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8095c5b56c7372d8a89568a5c7956f62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 11, 11, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_5393ef5558919bb609e69eea6fd1dc08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 11, 11, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_6892f9e4c6c48db827ff36b86160fb98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 21, 21, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_0c31fcdc568c4f7e20c3eb99c7832a04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 21, 21, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_ac46fedf8c9dd7403e84cd816dc95054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 23, 23, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_4281cc7c6c840edf10262b9cde75a7c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 23, 23, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_6523f0f88cd1b852a398256d6afc0cf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([5606, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_34c028487ed3c40b4b83d25b0352c4da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([5606, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_6523f0f88cd1b852a398256d6afc0cf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([5606, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_34c028487ed3c40b4b83d25b0352c4da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([5606, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_e352eb4bb6b247306d6d98688cccf45b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 11109, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b7c03b879f457eb72703fcffa5723081(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 11109, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_989d08469724e4838b11e136172db8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 19, 19, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_62b2c0c29dde3398dcf51196c07dca62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 19, 19, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_13c4b2bbf0aae91ad8469d88674de0cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 44, 44, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_06a0bcfecfba40c9e19d98754f8c1fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 44, 44, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_f919942e324e5aee6dd2fa2f853442a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1036, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_caa98b6bdaa6b3a5eb25fb5c54ae5bfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1036, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_f919942e324e5aee6dd2fa2f853442a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1036, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_caa98b6bdaa6b3a5eb25fb5c54ae5bfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1036, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_0031f0c0a493f1ee92c8d76700838811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 2100, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_4f4cafa5ba5c36c5a4bc8592495b6ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 2100, 4], dtype='int64').reshape([3]),
        ]


class PrimitiveOp_22e27089ef9e41b43f810763b98e9b24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc752558888ee72ca21e65f153279b8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e27089ef9e41b43f810763b98e9b24
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.0], [-0.8947368264198303], [-0.7894737124443054], [-0.6842105388641357], [-0.5789473652839661], [-0.4736842215061188], [-0.3684210479259491], [-0.2631579041481018], [-0.15789473056793213], [-0.05263157933950424], [0.05263157933950424], [0.15789473056793213], [0.2631579041481018], [0.3684210479259491], [0.4736842215061188], [0.5789473652839661], [0.6842105388641357], [0.7894737124443054], [0.8947368264198303], [1.0]], dtype='float32').reshape([20, 1]),
            paddle.to_tensor([20, 30], dtype='int32').reshape([2]),
        ]


class PrimitiveOp_692bbbab17f6cc00955b3df61900def9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b520450305d009e36fb59428094832e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692bbbab17f6cc00955b3df61900def9
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.0, -0.931034505367279, -0.8620689511299133, -0.7931034564971924, -0.7241379022598267, -0.6551724076271057, -0.5862069129943848, -0.517241358757019, -0.4482758641242981, -0.37931033968925476, -0.3103448152542114, -0.24137930572032928, -0.17241379618644714, -0.1034482792019844, -0.03448275849223137, 0.03448275849223137, 0.1034482792019844, 0.17241379618644714, 0.24137930572032928, 0.3103448152542114, 0.37931033968925476, 0.4482758641242981, 0.517241358757019, 0.5862069129943848, 0.6551724076271057, 0.7241379022598267, 0.7931034564971924, 0.8620689511299133, 0.931034505367279, 1.0]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([20, 30], dtype='int32').reshape([2]),
        ]


class PrimitiveOp_07395f6f2f872f0064de67c64e2f4b24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 20, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f50008877f7885e53afc4aa47948bc34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07395f6f2f872f0064de67c64e2f4b24
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 20, 30], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_f50008877f7885e53afc4aa47948bc34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07395f6f2f872f0064de67c64e2f4b24
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 20, 30], dtype='int64').reshape([4]),
        ]


class PrimitiveOp_ff696302730240262d0c26f825c0ae45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f28511b8b97ef55244383840e688adc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff696302730240262d0c26f825c0ae45
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b053b91422370dfac770cc66ecc07856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca331aa5f1d3f3574ff71efb635a13da
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2aba05af2b6fc75f7ce4e9dcde2537a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1260e1f7c3449d0b549f06437011aa17
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bbf509ac62f96af3feb683473c15a635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c819a798b53e114137e607f2f35d983
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d1489b83e20d976b74070b431b6d5f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_623261890a7385c161db7b0b6ecd69b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_607fef45e6bbfba88e0726042d959d65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_859868069b88f8596f1a0a7e5e83001a
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_02e51cfa063d50ce4fa5997f396d1434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4aa85cd9313e3f8e664abd2fe54781a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_10eb04559f77fa60f3c6ed6d9de97f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1809, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_3cb1f22fe2f262e3375e83c7f810dcb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1809, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_10eb04559f77fa60f3c6ed6d9de97f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1809, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_3cb1f22fe2f262e3375e83c7f810dcb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1809, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_720178596ae6c36e2d394a8b04dca308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3549, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5add0819a187947ec734746a7ed5bcb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3549, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f412f1f68b5b98566e86c46f21467d63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 12, 12, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_f3aa89b7ba8049bd5f4bf7055baabf2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 12, 12, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_b053b91422370dfac770cc66ecc07856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca331aa5f1d3f3574ff71efb635a13da
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2aba05af2b6fc75f7ce4e9dcde2537a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1260e1f7c3449d0b549f06437011aa17
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bbf509ac62f96af3feb683473c15a635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c819a798b53e114137e607f2f35d983
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d1489b83e20d976b74070b431b6d5f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_623261890a7385c161db7b0b6ecd69b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_607fef45e6bbfba88e0726042d959d65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_859868069b88f8596f1a0a7e5e83001a
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_02e51cfa063d50ce4fa5997f396d1434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4aa85cd9313e3f8e664abd2fe54781a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_607fef45e6bbfba88e0726042d959d65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_859868069b88f8596f1a0a7e5e83001a
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_02e51cfa063d50ce4fa5997f396d1434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4aa85cd9313e3f8e664abd2fe54781a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bbf509ac62f96af3feb683473c15a635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c819a798b53e114137e607f2f35d983
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d1489b83e20d976b74070b431b6d5f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_623261890a7385c161db7b0b6ecd69b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b053b91422370dfac770cc66ecc07856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca331aa5f1d3f3574ff71efb635a13da
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2aba05af2b6fc75f7ce4e9dcde2537a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1260e1f7c3449d0b549f06437011aa17
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_7f71645f3d2ad7f0e4ea49fedfb806a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32732f4bfd64e61263a6010d9d77fb2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f71645f3d2ad7f0e4ea49fedfb806a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[32.0], [96.0], [160.0], [224.0], [288.0], [352.0], [416.0], [480.0], [544.0], [608.0], [672.0], [736.0], [800.0], [864.0], [928.0], [992.0]], dtype='float32').reshape([16, 1]),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_70bb4b0f0c64ceb48e0a3b5fa3cbaba1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66e07b5da9a5b610759d0c2b24a21cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70bb4b0f0c64ceb48e0a3b5fa3cbaba1
    def get_inputs(self):
        return [
            paddle.to_tensor([[32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0]], dtype='float32').reshape([1, 16]),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_9478f4635cffdbe19478a10c6d823320(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7f7a706310dc38afb7c83affbcc4034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9478f4635cffdbe19478a10c6d823320
    def get_inputs(self):
        return [
            paddle.to_tensor([[64.0], [192.0], [320.0], [448.0], [576.0], [704.0], [832.0], [960.0]], dtype='float32').reshape([8, 1]),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_822ca17817cacfa4a5845e9c3d869015(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ca40b3cfb3f6ea2c86b22bedf8ac5a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_822ca17817cacfa4a5845e9c3d869015
    def get_inputs(self):
        return [
            paddle.to_tensor([[64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0]], dtype='float32').reshape([1, 8]),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_883c1681a5d606d0e3f6b27b70d85fbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 672], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fac9a16b6598f7f8cff15acf72f5caba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 672], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_40ac53095409212f7cc32c3036680b80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 46, 46, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_1c9a031b86f0ff555b6a8d6e20838ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 46, 46, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a6e198f13a4915a18efccc229c909053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 156], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d9f517c34a323febdbb0514fb31d0cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 156], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d06758742ab534fc8dae0294c837091d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_45e82ff12ef059b2ae4906f5beb64ab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 336], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_108d350266a171b0793b52a274a52362(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9304e745f55a828ea3948b9308856bff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_108d350266a171b0793b52a274a52362
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.0], [-0.8571428656578064], [-0.7142857313156128], [-0.5714285969734192], [-0.4285714328289032], [-0.2857142984867096], [-0.1428571492433548], [5.551115123125783e-17], [0.1428571492433548], [0.2857142984867096], [0.4285714328289032], [0.5714285969734192], [0.7142857313156128], [0.8571428656578064], [1.0]], dtype='float32').reshape([15, 1]),
            paddle.to_tensor([15, 25], dtype='int32').reshape([2]),
        ]


class PrimitiveOp_f7735c00b85e540e189bfc3d7cfc53d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b837486a72bab7796225a22b6ce1da08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7735c00b85e540e189bfc3d7cfc53d7
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0]], dtype='float32').reshape([1, 25]),
            paddle.to_tensor([15, 25], dtype='int32').reshape([2]),
        ]


class PrimitiveOp_b84cb87ef790691cf4ed2f88c7d513d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 15, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73dbbd61e3c78a9ee4d01b2e1f4f62fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b84cb87ef790691cf4ed2f88c7d513d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 15, 25], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_73dbbd61e3c78a9ee4d01b2e1f4f62fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b84cb87ef790691cf4ed2f88c7d513d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 15, 25], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_caf494325cb7785e9738e1ada3860292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 76, 76, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_9b40e6f1447139e8cf061e12cd6900b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 76, 76, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_bdefc5cfd8fe2418a8f694255c8eb6f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a9d90edbb5b23869d8b6302bbab54228(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_050343705fb2dba60c1fbb60e9d93cfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 872], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9de1b37dce0ba599a6b4154136cecc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 872], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fbfc8e7fc73fd2e8eef6705f1571e7aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 48, 48, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_1a2227b0a79e2f50ee38c91310c33424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 48, 48, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_9a76c2ff3c863a74caa70266c3132ac4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 24, 24, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_31ee92fa226dda05d76b5cd3bc60114c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 24, 24, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_050343705fb2dba60c1fbb60e9d93cfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 872], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9de1b37dce0ba599a6b4154136cecc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 872], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_651e8188ddc67f58aa6aa76f30150597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4179, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_bfb7ed54ca9a85975043a71c884d8725(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([4179, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_651e8188ddc67f58aa6aa76f30150597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4179, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_bfb7ed54ca9a85975043a71c884d8725(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([4179, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_6f7c87661a7dd945b61556f862395096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 8400, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5b087294c019160ba86716411dcd421f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 8400, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_521e5631875a8636a8f898705854a9ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 38, 38, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_77813509ff6b0c33d1fc3e1aa16d3719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 38, 38, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_eba7180482eac2e41371ecdbe37f5a34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 92], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c51ce8367bd3f87578a3f0974bf50aaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 92], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_99a1675c9a10f38e4148c6008c827b50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_861229d1ea67a7b7c7ac6390a132eff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99a1675c9a10f38e4148c6008c827b50
    def get_inputs(self):
        return [
            paddle.uniform([72, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([72, 72], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_e053c87e3736a6dc470282f43fcb0c35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b13780e3e0fb49644dde140162420b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e053c87e3736a6dc470282f43fcb0c35
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([72, 72], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_9a31eebf869e6dae4b9a0005b4ec59eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c29043cfa0dde9038b3aeeaf32aca0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a31eebf869e6dae4b9a0005b4ec59eb
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([36, 36], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_cc53d456e49996264c5363a23bee62bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f3743d748230ca5253651795450de2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc53d456e49996264c5363a23bee62bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([36, 36], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_3db7b08999dece65d233c42f0362c68d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51d0376b97569682b429bbd0e2d7c199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3db7b08999dece65d233c42f0362c68d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [32.0], [64.0], [96.0], [128.0], [160.0], [192.0], [224.0], [256.0], [288.0], [320.0], [352.0], [384.0], [416.0], [448.0], [480.0], [512.0], [544.0]], dtype='float32').reshape([18, 1]),
            paddle.to_tensor([18, 18], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_883991f7bd3af8703a438b9777ae00e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f297dfb74f8897f2de6159b1a020ea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883991f7bd3af8703a438b9777ae00e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0]], dtype='float32').reshape([1, 18]),
            paddle.to_tensor([18, 18], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_861229d1ea67a7b7c7ac6390a132eff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99a1675c9a10f38e4148c6008c827b50
    def get_inputs(self):
        return [
            paddle.uniform([72, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([72, 72], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0b13780e3e0fb49644dde140162420b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e053c87e3736a6dc470282f43fcb0c35
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([72, 72], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1c29043cfa0dde9038b3aeeaf32aca0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a31eebf869e6dae4b9a0005b4ec59eb
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([36, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8f3743d748230ca5253651795450de2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc53d456e49996264c5363a23bee62bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([36, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1124eb55d781afb9702747e6780eadca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3db7b08999dece65d233c42f0362c68d
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0], [464.0], [496.0], [528.0], [560.0]], dtype='float32').reshape([18, 1]),
            paddle.to_tensor([18, 18], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_239155edd35637331d47b19aab68e73f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883991f7bd3af8703a438b9777ae00e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0]], dtype='float32').reshape([1, 18]),
            paddle.to_tensor([18, 18], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bbf509ac62f96af3feb683473c15a635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c819a798b53e114137e607f2f35d983
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d1489b83e20d976b74070b431b6d5f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_623261890a7385c161db7b0b6ecd69b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b053b91422370dfac770cc66ecc07856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca331aa5f1d3f3574ff71efb635a13da
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2aba05af2b6fc75f7ce4e9dcde2537a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1260e1f7c3449d0b549f06437011aa17
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f8d4e83551fa628264f22294cf18feb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f71645f3d2ad7f0e4ea49fedfb806a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [32.0], [64.0], [96.0], [128.0], [160.0], [192.0], [224.0], [256.0], [288.0], [320.0], [352.0], [384.0], [416.0], [448.0], [480.0]], dtype='float32').reshape([16, 1]),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2e3160be92073fd54706bc170526cefa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70bb4b0f0c64ceb48e0a3b5fa3cbaba1
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0]], dtype='float32').reshape([1, 16]),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bbf509ac62f96af3feb683473c15a635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c819a798b53e114137e607f2f35d983
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d1489b83e20d976b74070b431b6d5f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_623261890a7385c161db7b0b6ecd69b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b053b91422370dfac770cc66ecc07856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca331aa5f1d3f3574ff71efb635a13da
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2aba05af2b6fc75f7ce4e9dcde2537a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1260e1f7c3449d0b549f06437011aa17
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d996270e7550f6f2414db764a1150e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f71645f3d2ad7f0e4ea49fedfb806a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0], [464.0], [496.0]], dtype='float32').reshape([16, 1]),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_11ebcb70166260664c0b415bf3783c47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70bb4b0f0c64ceb48e0a3b5fa3cbaba1
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0]], dtype='float32').reshape([1, 16]),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_4285b0f33a2edab7d4689060f4008602(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee7f2516182504d25ce83482d04dc3ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4285b0f33a2edab7d4689060f4008602
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.0], [-0.9130434989929199], [-0.8260869383811951], [-0.739130437374115], [-0.6521739363670349], [-0.5652173757553101], [-0.47826087474823], [-0.3913043439388275], [-0.30434781312942505], [-0.21739129722118378], [-0.1304347813129425], [-0.043478261679410934], [0.043478261679410934], [0.1304347813129425], [0.21739129722118378], [0.30434781312942505], [0.3913043439388275], [0.47826087474823], [0.5652173757553101], [0.6521739363670349], [0.739130437374115], [0.8260869383811951], [0.9130434989929199], [1.0]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([24, 36], dtype='int32').reshape([2]),
        ]


class PrimitiveOp_96730cedf2bf98b1b5fc10891c38eaa7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15f9802511e6d2518eec78dbe6df2e69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96730cedf2bf98b1b5fc10891c38eaa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([24, 36], dtype='int32').reshape([2]),
        ]


class PrimitiveOp_f59b1b6e82ac186de33de4cf0c5db94c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 24, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83f5a2636f46775035b22f7ceafc500a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f59b1b6e82ac186de33de4cf0c5db94c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 24, 36], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_83f5a2636f46775035b22f7ceafc500a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f59b1b6e82ac186de33de4cf0c5db94c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 24, 36], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_0dcd7036ef2b44d35b62fe8083c4ceab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_07e9a165e22a07353933a149f3d33d05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c8b9932f3aacf6efeb2e834caa532a3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4662, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_a5ab23a1ee6c04fe6464db7b9650c4cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([4662, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_c8b9932f3aacf6efeb2e834caa532a3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4662, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_a5ab23a1ee6c04fe6464db7b9650c4cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([4662, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_3c4ae96b27f36318d031083fb054582b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 9261, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_c068e9635539fcc38c1d611f0929336d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 9261, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_318b05213165357fbd52fd3a466febbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ee659b72a065320275f6e2db620120ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7bb4fd4f8146f4d235f4389dd10306eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3857, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_a304d4956a2e77f4be96305d81bf9dd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([3857, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_7bb4fd4f8146f4d235f4389dd10306eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3857, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_a304d4956a2e77f4be96305d81bf9dd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c231347eac57ccc7f7ecd36d187dd7
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([3857, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_b5737f5f42ab1c7629b96fe5e44229f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 7581, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_817653b73614c2c99f49a6e7335cf4c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f94225e3d64183cd49512b33e350fd4
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 7581, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_23dd152f5ee9138a11f0ffb0daa8de32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1248], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0f1ad348ae171bc8a80d5d192696f834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1248], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eba6a7680647335247a12e1131432761(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72c9193326671e924af6f80db707b3c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.0], [24.0], [40.0], [56.0], [72.0], [88.0], [104.0], [120.0], [136.0], [152.0], [168.0], [184.0], [200.0], [216.0], [232.0], [248.0], [264.0], [280.0], [296.0], [312.0], [328.0], [344.0], [360.0], [376.0]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([24, 24], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6da55859847c3815f0fdac80e202f742(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78e281f8df0f3999040c1aba884c6e06
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0]], dtype='float32').reshape([1, 24]),
            paddle.to_tensor([24, 24], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_553eaa956b0880d2cf9bd42c398ebfd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 120], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1e877bed1902d0355248a667a6bbb277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 120], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cd5fd8582b884f617af95d3d023225b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 22, 22, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_11ccf63b43d1dbc8f0d2062f85b853be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 22, 22, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_daec279770bd710d6c3288db3e89f464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3c413ccb493b35739c0d964a59d30642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b4c1f2595ade57c3e164839cda6a2e46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_df24aae0e36f8e413ad825750cb5ecf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1997699e0c8fd88c35221fefc2c3a6aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b8899c381f1439caae70ef076c20bf38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 336], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_0af98e7ada1f2ac4ca48a59f8f17c261(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e41e02d24bca529e8fb3fe3c0db9356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af98e7ada1f2ac4ca48a59f8f17c261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 32, 100, 2], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_030e861d01bbf70119d2a1964a57c810(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0af98e7ada1f2ac4ca48a59f8f17c261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 32, 100, 2], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_1997699e0c8fd88c35221fefc2c3a6aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b8899c381f1439caae70ef076c20bf38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_143cb78713bf926923ea9d5b7e93a803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 84, 84, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_0c214be839f5bb402cf549749d3a7e0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5900aceafa5b7c2d8700f8ae2942b047
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 84, 84, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_e2ca595e14c2f9f7de53431f9fbbe767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_62b1db730a6490c868b0c97d06fd4475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e73408102e3a7c6ce81ae8d86ab6fe7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 480], dtype='int64').reshape([2]),
        ]




if __name__ == '__main__':
    unittest.main()