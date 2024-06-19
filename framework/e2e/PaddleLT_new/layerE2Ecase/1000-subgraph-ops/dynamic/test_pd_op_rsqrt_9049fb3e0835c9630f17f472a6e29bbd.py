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



class PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.rsqrt(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee994fd9f07e38f1923623683d03828e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9321f15631e47eccafe41afeb0682d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3112d4c1cea037b6cb047ab0632d51b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a047e0ea4f84647fa7b0b2ed41d29b1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7973125439525ddc5fb6882592e37975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81701d5676f80480f1c4eaba93e3e915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5e88e5f67e63891edf3db258cffdeb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fe1f9c62731c0ce5703163f603d17d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_960db12c3d04798119a50fb312b77c1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f40a7b0f0bb9240ed7e317aa886a5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8faf4baa57e6cdc7a8ac5127b90a40e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dcb3232cb4a67d0a4b061e05c3571a71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f40a7b0f0bb9240ed7e317aa886a5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7973125439525ddc5fb6882592e37975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_94385eaeaefd0bb8fb791c0d70a743af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f40a7b0f0bb9240ed7e317aa886a5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab41e07427ada2d8cace6eccb0a58cc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8faf4baa57e6cdc7a8ac5127b90a40e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d3fbb607f631cd19e096b924614233be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8faf4baa57e6cdc7a8ac5127b90a40e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81701d5676f80480f1c4eaba93e3e915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4600638a539a48a8b11c3110beb6bdd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f40a7b0f0bb9240ed7e317aa886a5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_863b37b89d0d34894766616aa1ff1ddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab41e07427ada2d8cace6eccb0a58cc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8faf4baa57e6cdc7a8ac5127b90a40e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81701d5676f80480f1c4eaba93e3e915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8faf4baa57e6cdc7a8ac5127b90a40e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b4f665e79c13f4c385b3afc30850dd9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_69463948f380c4b3650b16b72a808c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_69463948f380c4b3650b16b72a808c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5892b9427f6452a5d1b23813313e918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8ae8014fd641ad61fa3c372aecfcc4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d72c344d1b6fb73db860f06bf6ea1c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d79dac38edddeba9a466436d2c16eaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d908dd01f276db382ed93a95c324e46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0c08cdc29eefaf2ee7591a245b5d2df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f4c02e85c2163de83dcc796b3de4393(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3112d4c1cea037b6cb047ab0632d51b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4600638a539a48a8b11c3110beb6bdd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f40a7b0f0bb9240ed7e317aa886a5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee994fd9f07e38f1923623683d03828e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_863b37b89d0d34894766616aa1ff1ddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f40a7b0f0bb9240ed7e317aa886a5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_695bc5e1a1a1bf696bc6ad9c11f00cee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_960db12c3d04798119a50fb312b77c1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5e88e5f67e63891edf3db258cffdeb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_23b650bc20066a2780ef6f317f162475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4dc48ecbd42e14e1d93e1209c480a56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8faf4baa57e6cdc7a8ac5127b90a40e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5f62b6cfc6e369ae2231c237fe1635bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8faf4baa57e6cdc7a8ac5127b90a40e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f40a7b0f0bb9240ed7e317aa886a5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_695bc5e1a1a1bf696bc6ad9c11f00cee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5e88e5f67e63891edf3db258cffdeb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f477a3d714fb348539b42516907126c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f9321f15631e47eccafe41afeb0682d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f40a7b0f0bb9240ed7e317aa886a5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c16d3631a9b967118ff37c610fe5e8fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ce8e1d06f53695cb8f8d6cb163cc0d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5e88e5f67e63891edf3db258cffdeb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab41e07427ada2d8cace6eccb0a58cc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5f62b6cfc6e369ae2231c237fe1635bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_960db12c3d04798119a50fb312b77c1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b4f665e79c13f4c385b3afc30850dd9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f40a7b0f0bb9240ed7e317aa886a5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9bdb14803d042096a555ef93073dd727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8ae8014fd641ad61fa3c372aecfcc4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8faf4baa57e6cdc7a8ac5127b90a40e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f40a7b0f0bb9240ed7e317aa886a5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81701d5676f80480f1c4eaba93e3e915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab41e07427ada2d8cace6eccb0a58cc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8faf4baa57e6cdc7a8ac5127b90a40e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8ae8014fd641ad61fa3c372aecfcc4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8eb787e9273bddbc711b99d0d0bfaca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d79dac38edddeba9a466436d2c16eaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4600638a539a48a8b11c3110beb6bdd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8faf4baa57e6cdc7a8ac5127b90a40e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee994fd9f07e38f1923623683d03828e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_23b650bc20066a2780ef6f317f162475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7973125439525ddc5fb6882592e37975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9000a041061f5b0ae0c40b4554beae2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7973125439525ddc5fb6882592e37975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3214df4407848fc6303bdd6f23f7acc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()