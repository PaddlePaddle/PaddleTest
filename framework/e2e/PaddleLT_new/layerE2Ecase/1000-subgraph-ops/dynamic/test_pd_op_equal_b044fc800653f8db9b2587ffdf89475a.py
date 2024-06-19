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



class PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b31eb6dd70b809526e63fd67133cd6e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92ddf0c441033e7f536ff47c4f834f80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[220968], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_be921cafe7f33076ae8f93d261954015(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_105eeb45b903a234849cac781dd2b721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be921cafe7f33076ae8f93d261954015
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a1dbc4cf07b05d574927dc0124285a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40535f2e07c8cb170e0953366f63ee43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[171888], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_50f74235838a71b54f59d29380575def(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be921cafe7f33076ae8f93d261954015
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_50f74235838a71b54f59d29380575def(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be921cafe7f33076ae8f93d261954015
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bfc4efe42dbb69f7d0dee9b1a122b65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be921cafe7f33076ae8f93d261954015
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1105e9552ad90ec7da64df7b5ebbc2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d3408ccfa458a5fbdcd817b2ce1cb100(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e542fc243fd9727368c80b00a6a0e300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3408ccfa458a5fbdcd817b2ce1cb100
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_07d3c764cc233a3e8e1dee397a195b09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be921cafe7f33076ae8f93d261954015
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_901e5957e277a832231998eea6ff6ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eaad3dc80a64d8d8ab29ff9ef47d2e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be921cafe7f33076ae8f93d261954015
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8aacbb4a58f97cf92595430f11cb8f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[185658], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c0afc3c06ab415db997f21f99af08ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2fbfd724ec2376c536221990b3c73448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d83824474197744a68f7deeecc273e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[86970], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_a865a54fab43355844625830697e7626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be921cafe7f33076ae8f93d261954015
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc88621df06e843668bbbf94069fd9e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be921cafe7f33076ae8f93d261954015
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_28c57851eaa743eddcb910960911a46e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_157edf2a5733d4d8daba161f2a2152b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_866889ab999b6b169a56c32154ba67e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1002], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_ca76f06bddfc677a3bbd8dd34c12148f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1002], dtype='int32'),
            paddle.to_tensor(-1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7578ac369670a878886cf2243143ed51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e542fc243fd9727368c80b00a6a0e300(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3408ccfa458a5fbdcd817b2ce1cb100
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffef22a3ab0f1ebf465e4fb84ff8dfeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[185691], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_ec6fad352713d019d4d345f70d48efa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3de3186f28e707b6172623ba13cb8825(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[123783], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_9b793e4cba7f6206f5fe1abe9474a2f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cfe39e20fe2bfae0874fc7c67cf19548(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[217413], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c9c731dc5ec4c5911a2b8bf29a2794fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[205923], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fe6204c24436692dd09fe8cbd506a8d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[242991], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_8917d167f6550d099e7fe0ececa9b9b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b0451d0c75a7566c75cf3ba695dff68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1027], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_69865fb874216d4c1b34cf303c13a74a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1027], dtype='int32'),
            paddle.to_tensor(-1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_85726172b709b33b8fa32720beffbc35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9d341255bc7238d76de13bac884b05f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9ce3646b8810e02318524d1f9fee56bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be921cafe7f33076ae8f93d261954015
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a085582b33bea492d7b5535a15b48707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[153450], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_1c8c8d13a8db07b0f702c7d3551233c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168b9e126be64d4cd1e7fd5df3a989ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c87552b7c12f64bda2fcdcc8454c0489(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[113061], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3c48ae116212f1a90c58aef0ea8a829f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2002], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_f59a23fb9b5c2bd93e0875447fbeacca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2002], dtype='int32'),
            paddle.to_tensor(-1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_f130604a49d9c48a43a9ae5d3ebaf380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1021], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3b5d3b338a68e4a753050516ffa0daf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45dc126167a71df6f3866f5c2f46c5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1021], dtype='int32'),
            paddle.to_tensor(-1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_e072f4f12a783dfb49f87d6ac259b0e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be921cafe7f33076ae8f93d261954015
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()