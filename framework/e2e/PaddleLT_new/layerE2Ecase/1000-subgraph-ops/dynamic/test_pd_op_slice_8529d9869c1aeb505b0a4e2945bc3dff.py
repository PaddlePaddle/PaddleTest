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



class PrimitiveOp_c1b0b571f9c2adee32b7729824698151(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d166ec672c6e498d724c238e58c3bcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa46d504627fad8b8f2427b7dbfffd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1473d090bdb42e4631ebe231354bbad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_153b218bd0185b56e821a273cc83ed84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6b3ab59a8f6d71b682f9de68a08dc1df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_da232fd767b12c6862173c2137aa3164(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4e36b934ab339a2e3ca35955212688c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6cc6a5e39b276558df5180b1b8f9d5de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_efcc8276dc399831cdb2fe202e363744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0bd1ad68576e4dfdd6cb488c95c97a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8b538f27cdd7e71783df52d5d9f262b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_052939de3e6b600830cf4ab6b2019532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b4e36b934ab339a2e3ca35955212688c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6cc6a5e39b276558df5180b1b8f9d5de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_efcc8276dc399831cdb2fe202e363744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_55a1c280050d842645d945a8e2ada64d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e70ddc49d84075347ca36cdd09318f31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([1524, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fead28af0385893bb608dea8dc8bb28f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([1524, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8bf26a99d48c5b852f0314f624ebc500(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([2340, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8fbff30b0652bb689998bed193b8428b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([2340, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b8d7cae215003cf297c4d7c67a619857(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_08e7262cfa46e884aefc5e4a61ff8710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e5700a498731b5ca1868d6e944e7d13a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1925bed963a2d3cb48a19e51e5f7a580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_167f610693fa8e75fa68009fd0af1c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c1dc6c44379870267a3fb8ae79cfc3a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7541754ca1f4f7e7900000702f6b7e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1dc6c44379870267a3fb8ae79cfc3a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_994cd21843df9658f0aa391cf13eba21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2f865af7c89b980da8a312765d1b68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d8fe00352dedbfb0f63a8d682ed7a770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2a557159f4849698befee003ec0da829(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7c0dee174a7f8649ec2ff299fd3c5c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9beab39fe2a122c5a348bdeb0544ee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c096b20c2c33af66fd52b41c7ba483d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06aba82194b71cbd392b955f8856399d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_755a7dd98475493129c571fe422e66c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_782b3b2cc0d3be729e9a145da28a38e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54a151c7936293d92b121ae16ac65bfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_85fbaf541ba59247cdad123a1f50c77d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_14390122803be2bc22125d081158100b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_076122d50bc2bf0ed6ac502c0ceee33a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea5a65d4d6609a4ecd63161197eadbcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d1380c8ca32546f575eeb514f113f836(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e35e788b8315988aec5546335af1ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c87171dcc4cd8c9407b61ba9e8bbbb66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15974219b536da9eedf33f7d0e887e35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c741f44d44ca0f9a28ea710ae87d033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef3ef22ebf9b8a8d8c61dde1acc085ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b0fa2f274116e2154f0eca0725ae7c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_115815a26cc99c19665f73f66b7a536b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5a17c20a97c9793379d96d62065ab25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_969d063f257079dfcf423f1b4fc2f7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b174a86185df34186bb00f6c13d75bb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37c39d952ff6537ecd2d1b5dbd00613e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7fc45726700d99d7c52df9d99e142589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_38fa7ff39454c54514e1646727fd23ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5311c921606fb00ff3db967636bd8ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3996bb1224a36fcca7290dea701a2012(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9fc84f49117b495d320e8a78e9278b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4ca2ebf0b46b8de4decd924638fa38a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5862d233af4f469a4a24b498d8ecf50a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3c8d54e1e72aebcc577f0c3ccbd841fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1ccab78bc7e35002d2fdd64b25ac8ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1680220920f0085b95f903d4d5c2e256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_791e7c0f447adff260a4b823134f6a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e710e549832051a5f8df8ac2c54675b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9cb1ed7720f99c764e3984c1f45b0010(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_756919ccec70dc1b2fe09b8d83208973(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_12777486c6b3604d4c4320816efc7957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d3f9e7f41b8db78cd936ff90442652de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_066a28c3b0da2c709751fc5b3317dcbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_edcaf048f46db60320eda90c820ab442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_477dd31deef8c0d9d15854c8188e923f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f431f39e317b7f6c6d748107a0daeff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f763b8a0d4a1122cbdd5fee7a95f9c8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6992556e3e3152683de1adc0ed261c40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_728a88c683a980618027e6d88b225000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a707419ce50bea40cd83aceb8e582142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3cb1e3cd60979436bb37d6b79181cb9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b784c4ca87c15b0e1d92b2db336f2418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_50482dae292ffe7bb5730f658aaac5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7dd924fc8fb5215bf362a3beb6db2337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([51], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d3b5cc498cb0c5cdc79864b4ee782299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([51], dtype='int64').reshape([1]),
            paddle.to_tensor([52], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2a4e5a2eef850d9ff2216063d6e65abb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([52], dtype='int64').reshape([1]),
            paddle.to_tensor([53], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c40fd8f972fe6b868d10f11d5f2544d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([53], dtype='int64').reshape([1]),
            paddle.to_tensor([54], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_13e54184e4aec4987c2e5ff50cd6f7ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([54], dtype='int64').reshape([1]),
            paddle.to_tensor([55], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b63d9398eea0aef5afa0c08db0fd6cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([55], dtype='int64').reshape([1]),
            paddle.to_tensor([56], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5096fcb84d98068d7f95c6313c5e671d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([56], dtype='int64').reshape([1]),
            paddle.to_tensor([57], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_33639338bbe1bafd20134016055b6611(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([57], dtype='int64').reshape([1]),
            paddle.to_tensor([58], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_11715e347e2d245e4e01aed6118c53b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([58], dtype='int64').reshape([1]),
            paddle.to_tensor([59], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e8e81b97f63ccf6cb929462d0f9a6d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([59], dtype='int64').reshape([1]),
            paddle.to_tensor([60], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_efc0f87a6d5c2447522881e527d5baf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([60], dtype='int64').reshape([1]),
            paddle.to_tensor([61], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5936226ce5c14830b5f44b951ab4b32d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([61], dtype='int64').reshape([1]),
            paddle.to_tensor([62], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a2b4676077d08e1e27bdc6c3538a36b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([62], dtype='int64').reshape([1]),
            paddle.to_tensor([63], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a4f1f1a843f37fe134444a707736f61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([63], dtype='int64').reshape([1]),
            paddle.to_tensor([64], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b66d39037122f7e492958b87e656ac79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([64], dtype='int64').reshape([1]),
            paddle.to_tensor([65], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7d9338a83500e6efba8b39dcd9e3b09e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([65], dtype='int64').reshape([1]),
            paddle.to_tensor([66], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7a61fc98394b15dfceb89e8d78d7b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([66], dtype='int64').reshape([1]),
            paddle.to_tensor([67], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18d7c30753ae7a49325db8c79b294d98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([67], dtype='int64').reshape([1]),
            paddle.to_tensor([68], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_87f447256bd71f7aea9c1ccfe99fe387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([68], dtype='int64').reshape([1]),
            paddle.to_tensor([69], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4dfc7f95492a2e82d72c86212df369d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([69], dtype='int64').reshape([1]),
            paddle.to_tensor([70], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad00bbbe7fb689dd8c72a0565e81fd2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([70], dtype='int64').reshape([1]),
            paddle.to_tensor([71], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eac8834ef4b8edcf1ad37c2ab5a9591c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([71], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e5ece939d0ff5a31ebd9b8bda7c15bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([73], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a478ec700104eaa29175a755a526f0cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([73], dtype='int64').reshape([1]),
            paddle.to_tensor([74], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d4cdb273a71d4ce0869d24a4cbf2ce16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([74], dtype='int64').reshape([1]),
            paddle.to_tensor([75], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b244afa012d890f93641d673909ab8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([75], dtype='int64').reshape([1]),
            paddle.to_tensor([76], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fef364eea51279595381e866c81d59f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([76], dtype='int64').reshape([1]),
            paddle.to_tensor([77], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a7f9be4408268d3658e3c3e15d0e8e94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([77], dtype='int64').reshape([1]),
            paddle.to_tensor([78], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f8555700bbb48748c9c515544a463375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([78], dtype='int64').reshape([1]),
            paddle.to_tensor([79], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_51a2b5d16325411730e2a3732669173e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([79], dtype='int64').reshape([1]),
            paddle.to_tensor([80], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4756bc66b9447a89d2cbe947d5830cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([80], dtype='int64').reshape([1]),
            paddle.to_tensor([81], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e7eb342f4f5ba1c5ebd98cb10f9f3849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([81], dtype='int64').reshape([1]),
            paddle.to_tensor([82], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_624e7823243802df77128b445b5275fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([82], dtype='int64').reshape([1]),
            paddle.to_tensor([83], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_adfe16d79a586d5b6b27e92d072a03d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([83], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8cbf211369e1673ead0538bd50dda726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([85], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a107b67cfe19eaf309e8ad8ed7580bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([85], dtype='int64').reshape([1]),
            paddle.to_tensor([86], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e469a6788dcd32a31cdd4e9287843341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([86], dtype='int64').reshape([1]),
            paddle.to_tensor([87], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d619e42fa7b6dd62a315bfde8ded8da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([87], dtype='int64').reshape([1]),
            paddle.to_tensor([88], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_280f08b211a3ce7cb8ecb3cca162e0ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([88], dtype='int64').reshape([1]),
            paddle.to_tensor([89], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_85ab69024fc6b8c8cac53c36a6a7c739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([89], dtype='int64').reshape([1]),
            paddle.to_tensor([90], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e711ec4bd6114b6917e5cbe888067e23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([90], dtype='int64').reshape([1]),
            paddle.to_tensor([91], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c96eadec49f1d225a84647466885c3ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([91], dtype='int64').reshape([1]),
            paddle.to_tensor([92], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_79d5ae46670b8f06b5c47cad4ab01c88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([92], dtype='int64').reshape([1]),
            paddle.to_tensor([93], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_702734f495458070755743a521942e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([93], dtype='int64').reshape([1]),
            paddle.to_tensor([94], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5afec703af1b259332e55e1354a3ea49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([94], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c86ce1fcb36e852772cc6087724be4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([96], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_787da53b9be22075dde5cff9916b2ae4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([96], dtype='int64').reshape([1]),
            paddle.to_tensor([97], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e42c22990411518c0afa1fd005c4cb38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([97], dtype='int64').reshape([1]),
            paddle.to_tensor([98], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7eb8672fc5d523346ad6a6c775159df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([98], dtype='int64').reshape([1]),
            paddle.to_tensor([99], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f647fddd0ed7a69d596f58724f37662e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([99], dtype='int64').reshape([1]),
            paddle.to_tensor([100], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_803e4b6a99f3f232f7f6edc354d95e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([100], dtype='int64').reshape([1]),
            paddle.to_tensor([101], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_71b512dba2111be47ca7ef87dde77e8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([101], dtype='int64').reshape([1]),
            paddle.to_tensor([102], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a17ca06350bb5a53cfa476a0f0ac1be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([102], dtype='int64').reshape([1]),
            paddle.to_tensor([103], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e2de922b3cb2da014935aa0c6fa78ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([103], dtype='int64').reshape([1]),
            paddle.to_tensor([104], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5caf77399847dd7cb64af015a12c31a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([104], dtype='int64').reshape([1]),
            paddle.to_tensor([105], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15e6c0189e03d52b882362f8c6e3fcfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([105], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_71c0d1a0a762fd0c1b14fa67bbdcd9a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([107], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_679b171c779f376e5ef7d3e68f9c046c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([107], dtype='int64').reshape([1]),
            paddle.to_tensor([108], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_89cb8e0c7feb2075086703f2485e6628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([108], dtype='int64').reshape([1]),
            paddle.to_tensor([109], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4013501c2613f3c17b6a25d389e081be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([109], dtype='int64').reshape([1]),
            paddle.to_tensor([110], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0525eaab5fd99904897e136931f71272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([110], dtype='int64').reshape([1]),
            paddle.to_tensor([111], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_513ca09359c7af163c5e18817a848c02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([111], dtype='int64').reshape([1]),
            paddle.to_tensor([112], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_12ce6b9accf98ae470b3f3c1e6757144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([112], dtype='int64').reshape([1]),
            paddle.to_tensor([113], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eefe1ef8a20f5a41d228293af9df5ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([113], dtype='int64').reshape([1]),
            paddle.to_tensor([114], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9def4a1e7aba361db46dfc2baf5427ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([114], dtype='int64').reshape([1]),
            paddle.to_tensor([115], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dc427bf04a14cc198a3692d03163dc49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([115], dtype='int64').reshape([1]),
            paddle.to_tensor([116], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d33476d775ef36ba5c4440889932b18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([116], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8222d4b4d57ac4b326369ac34568db89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([118], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a99da564457c5758b7f22bcd4c4d1a9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([118], dtype='int64').reshape([1]),
            paddle.to_tensor([119], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_492e19849b6b598fc8470afbe597cc3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([119], dtype='int64').reshape([1]),
            paddle.to_tensor([120], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a136861a53d19f048a1233b3fb12c717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([120], dtype='int64').reshape([1]),
            paddle.to_tensor([121], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c869f1423629f456162440c31e113c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([121], dtype='int64').reshape([1]),
            paddle.to_tensor([122], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_32bd3d4d2fbbf769e700ce1dfbf595fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([122], dtype='int64').reshape([1]),
            paddle.to_tensor([123], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18cf67de95ac4d5d6c52f4fc14322170(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([123], dtype='int64').reshape([1]),
            paddle.to_tensor([124], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0736c74ce76659ad76f4a1adfeb6651b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([124], dtype='int64').reshape([1]),
            paddle.to_tensor([125], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_418e4869a48695af1dd686203ada9f57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([125], dtype='int64').reshape([1]),
            paddle.to_tensor([126], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f603bc1b4c4813bd4fb976e6069e2d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([126], dtype='int64').reshape([1]),
            paddle.to_tensor([127], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b57980c3569f1a6ea31324f01cdbaed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([127], dtype='int64').reshape([1]),
            paddle.to_tensor([128], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a15e32f52734e8bd48a3cf281328e4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([128], dtype='int64').reshape([1]),
            paddle.to_tensor([129], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c4eaeb2e8cd80fa5ef4a7a85a52cce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([129], dtype='int64').reshape([1]),
            paddle.to_tensor([130], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4fbd18e33cd684128951de655f7d27ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([130], dtype='int64').reshape([1]),
            paddle.to_tensor([131], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15626b494840f9d58bbe8234c5510335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([131], dtype='int64').reshape([1]),
            paddle.to_tensor([132], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf95684f2644c324554df15bd0eff9b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([132], dtype='int64').reshape([1]),
            paddle.to_tensor([133], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6127e9fe02d600a77e135e883ac9244c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([133], dtype='int64').reshape([1]),
            paddle.to_tensor([134], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f1b8e7a205ca1e50f043f66eb973f97e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([134], dtype='int64').reshape([1]),
            paddle.to_tensor([135], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9a0ee97e21143f51c885ec0889eda8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([135], dtype='int64').reshape([1]),
            paddle.to_tensor([136], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43baa0d14846b7e021ee5b8db589b220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([136], dtype='int64').reshape([1]),
            paddle.to_tensor([137], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_608cd1cddd7c0d69d02602789ba16406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([137], dtype='int64').reshape([1]),
            paddle.to_tensor([138], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3729059d249327cacfe9ff8eec9ceed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([138], dtype='int64').reshape([1]),
            paddle.to_tensor([139], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84926a5e990aea97b0b33999b1a36256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([139], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0d1fabc74ee08b8684fde2bb0be4d97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([141], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_512ecbb0458f10593b1db09e737d9984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([141], dtype='int64').reshape([1]),
            paddle.to_tensor([142], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e65b251ed3223655e2e6b4b9be327894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([142], dtype='int64').reshape([1]),
            paddle.to_tensor([143], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d64082c6329968f6a70a35d21984cf35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([143], dtype='int64').reshape([1]),
            paddle.to_tensor([144], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6fdcdb41bc71222c801555beba76f4ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([144], dtype='int64').reshape([1]),
            paddle.to_tensor([145], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b5755cbab459a12d8fb5eac43811169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([145], dtype='int64').reshape([1]),
            paddle.to_tensor([146], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15a624204c6a0e4d1027bd9e8711f68d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([146], dtype='int64').reshape([1]),
            paddle.to_tensor([147], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3500c8442added9c9208bfd1c91794dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([147], dtype='int64').reshape([1]),
            paddle.to_tensor([148], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f622255476deffbeb7e35b64e1ce77bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([148], dtype='int64').reshape([1]),
            paddle.to_tensor([149], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4848611f41c244ffe3e0253a850562f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([149], dtype='int64').reshape([1]),
            paddle.to_tensor([150], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf4db992d3daa42457b8806aeda9b17a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([150], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c8ea12be45c669647ab8875bfa975b8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([152], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_75f6724929ad4847b46ef1dd44d4e556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([152], dtype='int64').reshape([1]),
            paddle.to_tensor([153], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d5715dfd44d97af924c161d11039eb77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([153], dtype='int64').reshape([1]),
            paddle.to_tensor([154], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_709088c9c7b61471190c289be9f84596(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([154], dtype='int64').reshape([1]),
            paddle.to_tensor([155], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7d1ef5d93b059d6b81db4d83a1ba3d90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([155], dtype='int64').reshape([1]),
            paddle.to_tensor([156], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0a358f79220b691318413c6e98af1e96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([156], dtype='int64').reshape([1]),
            paddle.to_tensor([157], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a0302ef153ae14ecf663b0a27c5c880e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([157], dtype='int64').reshape([1]),
            paddle.to_tensor([158], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e26424bcd6bb28bcbe29c37b8987732a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([158], dtype='int64').reshape([1]),
            paddle.to_tensor([159], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24464acc5686770a0027404096e7cbf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([159], dtype='int64').reshape([1]),
            paddle.to_tensor([160], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5819fba943d975c4f410c58a2188ef49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([160], dtype='int64').reshape([1]),
            paddle.to_tensor([161], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5c30cdf356cbb2076a42ffebbfedb507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([161], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2382f8e613e5c523f5a5c3158bc764ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([163], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e4c3267ed947d2d91dd5cf564c910088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([163], dtype='int64').reshape([1]),
            paddle.to_tensor([164], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ae7a0ef7ba5b8bc3b3e527e1309b6c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([164], dtype='int64').reshape([1]),
            paddle.to_tensor([165], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_39145117dd493cf7e7f1b993d6502be1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([165], dtype='int64').reshape([1]),
            paddle.to_tensor([166], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ee4c56a151a1d759bdd4084aa0be0fb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([166], dtype='int64').reshape([1]),
            paddle.to_tensor([167], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f8a71c3a237808b39a9889f606f3161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([167], dtype='int64').reshape([1]),
            paddle.to_tensor([168], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c90bb863163322e6b312eefd0a5333e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([168], dtype='int64').reshape([1]),
            paddle.to_tensor([169], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67701d46f136d08458e8923743328717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([169], dtype='int64').reshape([1]),
            paddle.to_tensor([170], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_94e0120355523d0b64a92336bf80194c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([170], dtype='int64').reshape([1]),
            paddle.to_tensor([171], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae18604944eee642865b685eb7bfb3b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([171], dtype='int64').reshape([1]),
            paddle.to_tensor([172], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cc58c7f689aad3341c5402d8f6cb9b35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([172], dtype='int64').reshape([1]),
            paddle.to_tensor([173], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_09b202b1b32a3fd678d5edd536077198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([173], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1eff8f9fd2dc4c1c2263e81a894e255e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([175], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9ff5a0e51442956d475bad5ae5eb4bb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([175], dtype='int64').reshape([1]),
            paddle.to_tensor([176], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e6e7a66923df107212b38200143e93ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([176], dtype='int64').reshape([1]),
            paddle.to_tensor([177], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db09d49e8e6af79204849f963d6033cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([177], dtype='int64').reshape([1]),
            paddle.to_tensor([178], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_849188715bc6a6ff40dbe077994248b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([178], dtype='int64').reshape([1]),
            paddle.to_tensor([179], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3aa1130cf24e81c02e2be2675f4a1a99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([179], dtype='int64').reshape([1]),
            paddle.to_tensor([180], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_765a74c294353b674561865e01316dd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([180], dtype='int64').reshape([1]),
            paddle.to_tensor([181], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e00c712b02641b7d3a1f102211c39bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([181], dtype='int64').reshape([1]),
            paddle.to_tensor([182], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c7c5ab188761c7660976a92b04ff4dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([182], dtype='int64').reshape([1]),
            paddle.to_tensor([183], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_91e8c932a4dec18e81cc6c6845d45e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([183], dtype='int64').reshape([1]),
            paddle.to_tensor([184], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a6d1722d07207295a2110057193de9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([184], dtype='int64').reshape([1]),
            paddle.to_tensor([185], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e981dc65fb17972ac7f78608a1db53f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([185], dtype='int64').reshape([1]),
            paddle.to_tensor([186], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5b62c1c66336e3faf9a87f4a910daf21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([186], dtype='int64').reshape([1]),
            paddle.to_tensor([187], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1fa8f9d689d0ec9509e714926e27663b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([187], dtype='int64').reshape([1]),
            paddle.to_tensor([188], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_439430cb4ae5980294409b8a8ade11bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([188], dtype='int64').reshape([1]),
            paddle.to_tensor([189], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8ae1df4f7a7d74b67f52c96cc6327cac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([189], dtype='int64').reshape([1]),
            paddle.to_tensor([190], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a70f0d87dd6610f689a125cbe3f6d36c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([190], dtype='int64').reshape([1]),
            paddle.to_tensor([191], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_19dd7d073bf690b0f6cf03fa63599244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([191], dtype='int64').reshape([1]),
            paddle.to_tensor([192], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e9359e9f33d2f595db3d9d0f94be6415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([192], dtype='int64').reshape([1]),
            paddle.to_tensor([193], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a97c2b51e136c7be58c755ff12447d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([193], dtype='int64').reshape([1]),
            paddle.to_tensor([194], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8e404ee5c8ecc1790581e65b577addd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([194], dtype='int64').reshape([1]),
            paddle.to_tensor([195], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9d44fb5ad8301376a897a0261592e0c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([195], dtype='int64').reshape([1]),
            paddle.to_tensor([196], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d2e88242d6a00ecc9455754923a5e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c31af418577084129a1ce33ad645c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0fed8e79066b7fb54b96e16562f55519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_947467a08ad15cbbee76b357f40c7b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0287e75229855e44510d3ed7d62a19ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3579a22a2165b5cb9564d3a08061490c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd38b83a46432a3f1892d5fbd9cef9ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ba6ce166c3755c18420f85354e724c79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_578efc777bb292a9384d3bed64d73fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7fc1cdac3cfba1bb3e518279bd64b24e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84e8c24c924767c6eb1d0559f1aa03dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_34511f51d66211c9fd8284b965f603e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6432aa4a4596983e17e2648b50653550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f1549adeeda8280588e3484cacba120(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_50e34ba0f19d780995dc2634088bcba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_62e630428b99dcead4f67c31a96e64d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a863f945630fd747534eee1988ae949b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84e8c24c924767c6eb1d0559f1aa03dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_34511f51d66211c9fd8284b965f603e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d724aadafe2d0006a02ef86d7a1a431e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a30640b238d8fee4b4a63cf3f304510e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_56969192eb38c889021b343a3e455c28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9f7a78235379e86db22c384e8c9b2c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b344f2ca33d47bcd87ca65e836b8457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9659325a86d93db988d8fb6cccdbb756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00e9ac64c7ed3e4420a2857e9132836c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8dfbc721abef34a842e55b30c8297650(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6432aa4a4596983e17e2648b50653550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0f1549adeeda8280588e3484cacba120(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 64, 128, 256], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_266ac6d4ae68446fbe97d27eebef11bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_231afa21e0f1a1a8ff0a066675fbf553(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 192, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f8f83e14bea6007554b4917ef7df2f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5cac114810e586c7607ba155e6bbc2a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d7c90369453ceca251ff6596e7ed0ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_99f0c25f8b81c11fa98bcd5fe00f90f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6a394dab97c574a514bb78b5c470e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 4312, 4, 16, 6], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1aba18082853e4c22e5ceddcf5096eee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 4312, 4, 16, 6], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a1d55c596bf1c8bb94ca04610ef7397f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c864df861f04e26becfecf67123eb854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8ca5638d42421565fb2df289a09ba568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed31b0817024c974a1ccfaf2730a3d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1dc6c44379870267a3fb8ae79cfc3a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0e9f39bfd80f160a542bdfad7dcfee99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7cfb2aaa789115291cface4fa860330a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_05b68eb8e21c447897474a45e67c24cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed31b0817024c974a1ccfaf2730a3d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1dc6c44379870267a3fb8ae79cfc3a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_65aab3f87e18c7e4fb448b7b3f711594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([2047, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f88363e627a14b422cc8a079ca977309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([2047, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b8d7cae215003cf297c4d7c67a619857(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_08e7262cfa46e884aefc5e4a61ff8710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e5700a498731b5ca1868d6e944e7d13a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_833f9250a3f496fe8d03a432680cf943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6f1efe88f1e10c4d1b2662f5a7a1977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_adccb415abedc971fe6176dc0796b558(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5c79d2bf64572ebce43c89d783705fab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fbaa7187d6c4f6847ba76ccabc2ea5a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f73bb76f49be41a53785eae95f4c91af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea0c52f15bc31c253e037e3936c59faa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8ce63271f7bb2141ed31e547809e3366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3f5d8bc09e66e8a25e3e241b7e2a78c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b16e8eb5c303559282edf10757a1cdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6928b0c1a1a59455175976f9ef4cef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_51e7576469d11cba87b1122f286e3814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_071a3cc4f628042c5ed5ff853481af30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d942e8ae1aee301fdcf8eacf681d7264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4efcd21778d1f954a383ce7eee0394e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3657fdfe6494726f0f59f10406c61079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a7f48cc78088541bafc2e73a644d105c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_96015a2d5be6b6f490ea45b56110c5b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e8ce781a839ff1a7bb761ddef2378fbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93f4592029fd1dbfb63592435fa6c976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e2f8901a8f746107fa55a0718468c29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5240447ef357a409e0c9ea7b73339b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4fd72ac477e6f424dd9c68eddce6e8d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7bded0eaf0e2b046eaf1ced20b87088b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ee83a1cccb4c4ac9c6997dbcdfdae3fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_54ce42d5b7897992e5981953d54fc7b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fe4be8f4aa70c9004da5a05ca2998567(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a512c9d69df4635aff2f9a457ceeaeb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e6e321a21b85551d39f18ba70004e325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_69a15dfed3f4cdc593a1a8c5c0c5aeb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4baac5d906de1c8e0c33441969ff6aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f3a1607a2609fbf55a4914d527d59bf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de147bafd9686e309c8b3a303e1fb9b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6cf80e547e29e188f746f60a2441a45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_85a075dce2e9b59bc36e966e2eb8aa70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1ce420da84247bf43b910e5c75ee14cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2f84b7bd39ffe8d563b7dd94285b4c50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b75fb44caa650328fb0d52c9efa5bc6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28f2d33627d8713e9031b9802b523b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_58263b417ae58dd2109ddbe2fdae05dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1677a72d1b6bfe79914d381e1d3dd803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3459648debdd29d0ae8ab14e41099590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5cfd03df5a29451d2bbfa1fe1ce4c3ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa199a8792bc59209bea3530ed2b0403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e4e88a3080f903a92b731663b8474531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_87ab91e19a18cccfb9a3f91a1d50000d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a7fa725fb2590114d790ae54ec6fa339(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6363018f4febebc1599660a2ba2845af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_96dcd84b1b74fb76ebae09d805b90761(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24fe7bb522531c95776b51975314069d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6153fa0aea9604049b6dc87720babc27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0bb39b051aa856e1dab427841e08f6a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_459d7e2758978e5bdf8a487496bf2395(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_edf3fee725b94d2fcfae35cb99e146e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_008d954cb529c4236a31e9f752e00c37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e24ca639c53ae4414902b9468d2854a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 256, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6146292651abea81a8d931c4c7b360aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7bd2a8bb6834fc377b2c81233031bb96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93039301bb933ce9990f19eb97226ae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e3323dd497389488a7e3838fc6362fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dda07d0d56a0cb977976262a7d371b16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_42efbae9309298bb871529d7994244cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_499d3c7349e7e3cb624a7cb7ad57eb0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_16de94558eab377b780e06f7264c3e46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6b1b482300ce46bf97a56d83d90b4e95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_386fd3a039d00ad0d838ecde9910201f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0d47be68ea380f6d1309012d78139de7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5c7db30525f4f5b97671cc48138d24c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3acba6262093d184d5a596794a28595e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dcc2266911153aeed1de17f177a582f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9601be76a3ee78a5e2dd845b6a72c06f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1195d82934e2bdcaaee090f9d4002095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_deed659d81d79eddf43189f83d4bc9cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([1813, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18b267449ed3c971b671f4008e4524dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([1813, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28823b9e7faa2c50ccdfbdb4774dbb66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dad7cdcfaf5d51b89719ebb768390fc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ad6768dfb4ac7203c539b5184f41b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_59998efdb521202935a1a2018668f6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_41e8f12525c2d3603d2cc7045860b632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b2b060dd44460a71ca3081a91ef6e4db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([3061, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a1f26d0f068677d645d5fcecc9b8740b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([3061, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2194f308789bbc02347f0cf669700b2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_943441f9ee101bcfb7fa7e1cd2395265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 1024, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00e9ac64c7ed3e4420a2857e9132836c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8dfbc721abef34a842e55b30c8297650(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_680a455ed3e67e6e68d6c2204e67e88f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ac3d8fd6f254aa7594e1f4821a248f48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_619708a9051d1c6f65ed87be4be11806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_21f90660c2081d0a662df7a60a9fa168(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_deb1f255f1ad5dd09bf486bf0eed6008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9d107e521abed74e82c8b61b4141b789(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3080bf7a272db9ace5f4d2288dd26160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e1f470f39487439658774f85828ca637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_85e8d10b7c9d6f001eef3e0759176748(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a8c30c2c5e8f2c8d8eef93d5ead3bf6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7fb815b2c891fd5247f269bb76d9a1a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01079878f21f5af49ca74ff38045751e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8fdcbf0375d4740c02435778242317c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d1bdb298ce0dafeb92fb88831043a50e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_04fdd9f08fd13256c9c1988afe867f6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_11e2cb053256d1d3dcf7614047c2627b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2f49e6919c6832b2ddf023bf38e3d5b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1abb96111c3f5373828828d31904bee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_31a8f0928ab085397f764e4430f2164c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1d02eb3976a47a3943b627d762a7ef7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0a4a67e6d330eca8fcfe4568808bbae3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a16be37d3dfbe9696299dfe8462cc8ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e5cbf19ef91ee8ee300c40379fa1865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b7a7c8757649a6478a9bd3d97eb0000a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8cee3415d99820f054e8da49162254d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa6990c58c889a0b51b13b14f419617b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_886d8ecfe3c8ca8d965b36bd7c454dde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd880c0c3679e9c93da9a49d592da377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15d830688f346b082bd40380df09db4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d4f36bf562ab18238925e5c8080b4d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_30d12facd466af78f1010e074a6e1063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dcbf60add73077f314d7978e253ec1b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c70f6345c9cb6dd091b3c6d991f4f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46231275e8568e92065cc31bdec7c409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_881430a915f53e9a4e538dbad61aa95d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_12e130a833e21953bbe0c71754c1d8ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_73d7288cf8fbb14283bb7c74b8f00709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d2deef2fc95cdd729678bb668cbed5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f0f80950a5cfb1c1b0fd8cc43036ecb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1d3a6d93651483fe84684db8c3c1534e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_16e14f0145dd9992857cdeeda21e85c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3b03dc4d72ec2df5132c6bb7fce8672a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_35d53e2d38482056901d2743d371483c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d25af9ffcf05807c00931a3eb09113fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c85298fd5628f336628d18183fe6492b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_356c5409c1316a0479ccd088905d73d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f900f000d2a32009c07ce4ae75f3ab38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([41], dtype='int64').reshape([1]),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_167069ff5c0ff5782d57496919ab4a82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([42], dtype='int64').reshape([1]),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24fa2928255acb37c46f8d740849f2cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([43], dtype='int64').reshape([1]),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d32055f16ced9579f0a2820fa6e71200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b8174e7d4a9dd83ebf97e844ac863235(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([45], dtype='int64').reshape([1]),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_52acf72c6898b21f7729ce52152ffd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([46], dtype='int64').reshape([1]),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ccf8db74e9d53f70bcd55995baad98a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([47], dtype='int64').reshape([1]),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_24a62ec42bdbde3176300e51bca02274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([48], dtype='int64').reshape([1]),
            paddle.to_tensor([49], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_143552cbc33bb6f668d676de3180ec24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06e01a9ab98ff903b1ff92a355f715a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15d3433429adf42d48390a15903c16dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd6277ee13835399ac92102d0eb5c6e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a238ec617d84b10bcaa00b9ca064de16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([2062, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_78d47f97a5d9d7e002f39222ef903d46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([2062, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9fd4bf459b7073c00c9274445f21624d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 22, 6, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7d6283a8ff27dca75f84fffe3957028d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 22, 6, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b56a68e5bd2a6d02751b23cadb1e33ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b3f947d2bbf752bc077f24bd903bc928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6e2d5d534601969364c700cb0d373e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dc9ab0a494ebaf6de10ad03a3d4065e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43ad36a47fe8953ccea79d14a54f2316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c008bfa8f0f225d64ed2691aa293dbfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1db50e4f6e24bd1b1560ba28c63458c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0626042987ed439572ae7151a967ea5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_efe7e645ddd448b4f9f4166c2d521bc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0365d96d915f163997b375151e4bd87e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d01e03084c145cdb58d000e35de85a8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e9dccea55284af698d10c325b7534ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f07143f637ef3872d05f888a7e3157c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_524b4d62742d6cc3658a9e0050a41922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cfdc012631ea1bc97ba8b27de53eb263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_98034474fd321a173e1d0b0d7a401ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a557159f4849698befee003ec0da829
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b895be6edbe377b7359699d3f49dccd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_34b2af2038cd5e03d34a5eb83dd86428(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_541949497b9bca41ff9b6d6ea72f9ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a04c55a6fddcf9e381a23aeda5fc9375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e766efcb6fcc55339d9e4ba7cb30115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_96b60d97effad1a2073b20691c66d323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebde116a205f47f8a66622e34485a7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f222e4d3bba2017da376e8753189c87d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3d166ec672c6e498d724c238e58c3bcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa46d504627fad8b8f2427b7dbfffd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d7c90369453ceca251ff6596e7ed0ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_99f0c25f8b81c11fa98bcd5fe00f90f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d2e88242d6a00ecc9455754923a5e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c31af418577084129a1ce33ad645c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d2e88242d6a00ecc9455754923a5e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c31af418577084129a1ce33ad645c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2d2e88242d6a00ecc9455754923a5e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c31af418577084129a1ce33ad645c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_942a2197e2635c0a7c43f4e433f412f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c3408c3810ee406853d692aec0e02a3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_57d4707be781425ae96da7b33e926604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d825f0d5a481dcefed6a6f4be1f2bf19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f07fe0536d7b4034d3625b70ce273f32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_40e1beb1ee1dd61a6a3fc443d8541fb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_50024b2a2c4726781aa7e058898168c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_02cb7032c5ad40f6369987638d3562d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f8e6785115f2b5b10270d794f33040e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3122690eb8ddc28b7c7a1dda4de39abe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d79e94760e322f573bb4f0b198a62260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebde116a205f47f8a66622e34485a7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f222e4d3bba2017da376e8753189c87d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebde116a205f47f8a66622e34485a7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f222e4d3bba2017da376e8753189c87d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ebde116a205f47f8a66622e34485a7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f222e4d3bba2017da376e8753189c87d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9fbe64a02c84fab3b9c0af6da9caa3e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9d17bfa1a26a37d3af2b4a411d531c28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 2048, 64, 64], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e6aab0f153e053556737ac174a664b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([5526, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1935e555fe2a602a2e3c9da1f18708c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([5526, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a1d55c596bf1c8bb94ca04610ef7397f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c864df861f04e26becfecf67123eb854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8ca5638d42421565fb2df289a09ba568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3579a22a2165b5cb9564d3a08061490c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bd38b83a46432a3f1892d5fbd9cef9ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ba6ce166c3755c18420f85354e724c79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4897eac8ce79ed5648c9b30f23e2c9ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([1071, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0df79b230294098d625c34908e1bcbdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([1071, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15d3433429adf42d48390a15903c16dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cd6277ee13835399ac92102d0eb5c6e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 128, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_680a455ed3e67e6e68d6c2204e67e88f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ac3d8fd6f254aa7594e1f4821a248f48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_31f65f94f9516e1eea294c9e230b46d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_761bdbc5a2adc5c0d2188ee26e7c20b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 512, 97, 97], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ac06c16672e1920452166155e66bea4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([1760, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_80793000542fe2f99a9a63c5feb4dfb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([1760, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bb6debfde06ff96ecbde8b23f7b1fb84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a809ef95a99ba429a076cb100e4d013b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_48d47ff2fa702545631ca6914db9fcf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0fed8e79066b7fb54b96e16562f55519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_947467a08ad15cbbee76b357f40c7b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0287e75229855e44510d3ed7d62a19ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4c151920a25fa8c9e569c0b561fa0063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 10, 6, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2487cb81cdd8d1c6bf2f9f66bab2ae02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 10, 6, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a398dad57114c4fb7c785709cfba02ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a146890f5eed4b57ec722daeb0a37cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_83025ae7f4846fed1dc9fec10fdcff7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d724aadafe2d0006a02ef86d7a1a431e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a30640b238d8fee4b4a63cf3f304510e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_56969192eb38c889021b343a3e455c28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [2], input_1, input_2, [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14adf259e9195909230e834511c9eb47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.3889973759651184, 0.21859318017959595, 0.1871953010559082, -0.3016904592514038]], [[-0.05222040414810181, -0.11547282338142395, 0.0767585039138794, 0.04326122999191284]], [[0.08656024932861328, 0.30508702993392944, -0.4223214387893677, -0.16104766726493835]], [[0.31698358058929443, -0.403150737285614, 0.46750426292419434, -0.023694336414337158]], [[0.09362101554870605, -0.3219117522239685, -0.3059464693069458, 0.263411283493042]], [[0.28389525413513184, 0.22421681880950928, -0.21983292698860168, -0.1024852991104126]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_61dbe0891682fde2c032b6dd1ceaa80c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.3889973759651184, 0.21859318017959595, 0.1871953010559082, -0.3016904592514038]], [[-0.05222040414810181, -0.11547282338142395, 0.0767585039138794, 0.04326122999191284]], [[0.08656024932861328, 0.30508702993392944, -0.4223214387893677, -0.16104766726493835]], [[0.31698358058929443, -0.403150737285614, 0.46750426292419434, -0.023694336414337158]], [[0.09362101554870605, -0.3219117522239685, -0.3059464693069458, 0.263411283493042]], [[0.28389525413513184, 0.22421681880950928, -0.21983292698860168, -0.1024852991104126]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_69d3fc6551bfe99e871122f65d85695f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.3889973759651184, 0.21859318017959595, 0.1871953010559082, -0.3016904592514038]], [[-0.05222040414810181, -0.11547282338142395, 0.0767585039138794, 0.04326122999191284]], [[0.08656024932861328, 0.30508702993392944, -0.4223214387893677, -0.16104766726493835]], [[0.31698358058929443, -0.403150737285614, 0.46750426292419434, -0.023694336414337158]], [[0.09362101554870605, -0.3219117522239685, -0.3059464693069458, 0.263411283493042]], [[0.28389525413513184, 0.22421681880950928, -0.21983292698860168, -0.1024852991104126]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c58e6dd07f8818b8426989766d914beb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.3889973759651184, 0.21859318017959595, 0.1871953010559082, -0.3016904592514038]], [[-0.05222040414810181, -0.11547282338142395, 0.0767585039138794, 0.04326122999191284]], [[0.08656024932861328, 0.30508702993392944, -0.4223214387893677, -0.16104766726493835]], [[0.31698358058929443, -0.403150737285614, 0.46750426292419434, -0.023694336414337158]], [[0.09362101554870605, -0.3219117522239685, -0.3059464693069458, 0.263411283493042]], [[0.28389525413513184, 0.22421681880950928, -0.21983292698860168, -0.1024852991104126]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_619708a9051d1c6f65ed87be4be11806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_21f90660c2081d0a662df7a60a9fa168(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_deb1f255f1ad5dd09bf486bf0eed6008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f7541754ca1f4f7e7900000702f6b7e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1dc6c44379870267a3fb8ae79cfc3a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f8f83e14bea6007554b4917ef7df2f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5cac114810e586c7607ba155e6bbc2a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd3458050ff600dbf42f4db37322d597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([4204, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6910bd33f3cfb44ad1ea4d56fd671976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([4204, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_363128aa02166c5873f280d80661a9f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5881022ed9281acf261d44a195b9d5fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63996e3ae5bcf5795cb8460501a933c2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 256, 64, 128], dtype='int32').reshape([4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3541fadbf523622fb203733a3d6d0a95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef8c3368806811065d6399e1eb63210d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_adccb415abedc971fe6176dc0796b558(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5c79d2bf64572ebce43c89d783705fab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fbaa7187d6c4f6847ba76ccabc2ea5a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f72b73508f93243b44a8eff48890da83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([4680, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d11005ef011d07a97ec80bec7241b11b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([4680, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_77bb09931073123f7a0adf9e22254211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd8b09a32ef6e15b80e8ef22c0c19b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 8, 512, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3541fadbf523622fb203733a3d6d0a95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef8c3368806811065d6399e1eb63210d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_be735f5e453ef41042dc7c621c5615e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.slice(input_0, [0], input_1, input_2, [1], [0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9970ad5cdc16992210108a3f20ba550c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be735f5e453ef41042dc7c621c5615e7
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_97094cb729492fcd0844f1c2c54992f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be735f5e453ef41042dc7c621c5615e7
    def get_inputs(self):
        return [
            paddle.to_tensor([4, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3cd13eb88d23dcedf688fbd9029275fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([3778, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0ab7ae12a18861d0982b41f4a1da088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1c280050d842645d945a8e2ada64d
    def get_inputs(self):
        return [
            paddle.to_tensor([3778, 4], dtype='int32').reshape([2]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1473d090bdb42e4631ebe231354bbad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_153b218bd0185b56e821a273cc83ed84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6b3ab59a8f6d71b682f9de68a08dc1df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_52552697008e8e55c8894e9d61a0dffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d8c52a3f3757aa849feb3a9424b17811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_05e00da291eed759cff98f0874c39b84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([3, 1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_143552cbc33bb6f668d676de3180ec24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_06e01a9ab98ff903b1ff92a355f715a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 11, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0969e5cc44b975212fa1bd1805264ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1960, 4, 16, 6], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_96e2f4f44f9d824c0a3f87f254d8c2cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b0b571f9c2adee32b7729824698151
    def get_inputs(self):
        return [
            paddle.uniform([2, 1960, 4, 16, 6], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9b50385ef821b9cf814a97cc76f124c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.41676831245422363, 0.42071056365966797, -0.3246951699256897, -0.3679172992706299]], [[-0.29034411907196045, -0.3155045509338379, -0.292475163936615, -0.4908938407897949]], [[-0.360625684261322, -0.24923381209373474, 0.20421487092971802, 0.18880033493041992]], [[-0.17076244950294495, 0.24133849143981934, 0.21846681833267212, 0.4726373553276062]], [[-0.10002967715263367, 0.10318785905838013, -0.2251012921333313, -0.2904278039932251]], [[-0.46427422761917114, -0.05146479606628418, 0.03535884618759155, 0.09679967164993286]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a65ca030b40b76486f2d433154596523(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.41676831245422363, 0.42071056365966797, -0.3246951699256897, -0.3679172992706299]], [[-0.29034411907196045, -0.3155045509338379, -0.292475163936615, -0.4908938407897949]], [[-0.360625684261322, -0.24923381209373474, 0.20421487092971802, 0.18880033493041992]], [[-0.17076244950294495, 0.24133849143981934, 0.21846681833267212, 0.4726373553276062]], [[-0.10002967715263367, 0.10318785905838013, -0.2251012921333313, -0.2904278039932251]], [[-0.46427422761917114, -0.05146479606628418, 0.03535884618759155, 0.09679967164993286]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e6f578eee5c3652ded3765eec02b215b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.41676831245422363, 0.42071056365966797, -0.3246951699256897, -0.3679172992706299]], [[-0.29034411907196045, -0.3155045509338379, -0.292475163936615, -0.4908938407897949]], [[-0.360625684261322, -0.24923381209373474, 0.20421487092971802, 0.18880033493041992]], [[-0.17076244950294495, 0.24133849143981934, 0.21846681833267212, 0.4726373553276062]], [[-0.10002967715263367, 0.10318785905838013, -0.2251012921333313, -0.2904278039932251]], [[-0.46427422761917114, -0.05146479606628418, 0.03535884618759155, 0.09679967164993286]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_53c91ae4c7f7d1b61e314e8db8e364f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1dddab2e841cc3aff6500aabb30c9c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.41676831245422363, 0.42071056365966797, -0.3246951699256897, -0.3679172992706299]], [[-0.29034411907196045, -0.3155045509338379, -0.292475163936615, -0.4908938407897949]], [[-0.360625684261322, -0.24923381209373474, 0.20421487092971802, 0.18880033493041992]], [[-0.17076244950294495, 0.24133849143981934, 0.21846681833267212, 0.4726373553276062]], [[-0.10002967715263367, 0.10318785905838013, -0.2251012921333313, -0.2904278039932251]], [[-0.46427422761917114, -0.05146479606628418, 0.03535884618759155, 0.09679967164993286]]], dtype='float32').reshape([6, 1, 4]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_40e1beb1ee1dd61a6a3fc443d8541fb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_50024b2a2c4726781aa7e058898168c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_02cb7032c5ad40f6369987638d3562d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da232fd767b12c6862173c2137aa3164
    def get_inputs(self):
        return [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()