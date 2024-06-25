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
        return True
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



class PrimitiveOp_abe97bd705ac58eeb73fd18d9c007280(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80023cb010b470833506c92e7baf741f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abe97bd705ac58eeb73fd18d9c007280
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_94910f8ee260fad0a119d73990fddcd3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 24, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3442b95634c26b5c28df1cc2ee9b1f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94910f8ee260fad0a119d73990fddcd3
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fd01578e491d89190ede399f0322bdf5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55e70ae9d1d5524acfff38703759a94e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd01578e491d89190ede399f0322bdf5
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_555632345c2228b85cb7f93d7106c9f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 84, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5039e9454db939ddbce69eaea6e1c9e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_555632345c2228b85cb7f93d7106c9f0
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd7639611dec971c0f7a68b155ec5c72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd625d43bad8fda45ca41ffb01fba60c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd7639611dec971c0f7a68b155ec5c72
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1fe8d1b90d829d23adb8850b99451ca6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1912cefc4f5d96cb0dc5df79ed0143d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fe8d1b90d829d23adb8850b99451ca6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_84a68563f1b9d896c0d75c4b1a663f28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b6139ac6a6efd2e3f5c8bf0d510f39a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84a68563f1b9d896c0d75c4b1a663f28
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_784739d2851e4e10c40cb833f72a65c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3e67089058f7f932304c0a9438bea37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_784739d2851e4e10c40cb833f72a65c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_da99893753fbde62f75a437c84e711f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_399f14868ef8ad8d347d71997800fa50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da99893753fbde62f75a437c84e711f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ab46d2cf5dbf91ecd095091b12934cbd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 15, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc3cde6ab13325b7b55b0a2c1b331a83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab46d2cf5dbf91ecd095091b12934cbd
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5a56e73239314b617969b0ce8779443f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 112, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69dcf519d8fe23d6825b05c1cbbfc8b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a56e73239314b617969b0ce8779443f
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_446526674c1e45fbbe5dcaef803923c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f19322093840d0c1f174ac5a571ff003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_446526674c1e45fbbe5dcaef803923c0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.2199201583862305]], [[2.3082380294799805]], [[2.558354377746582]], [[2.0145928859710693]], [[3.115633249282837]], [[2.7547245025634766]], [[2.1783387660980225]], [[2.085667371749878]], [[2.227307081222534]], [[2.463222026824951]], [[2.339259624481201]], [[2.1679656505584717]], [[2.884054183959961]], [[2.9636123180389404]], [[2.4699795246124268]], [[1.4818334579467773]], [[2.742969036102295]], [[2.724429130554199]], [[2.263493776321411]], [[2.4747672080993652]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class PrimitiveOp_13203893eee5872df8e2a5c90635fb7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1aa0a12f2813bafc0b06141626c762f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13203893eee5872df8e2a5c90635fb7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a5d306cd00dbd67f68032f75eba9b97f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_281563c7ce860316cfe503762d6af8af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5d306cd00dbd67f68032f75eba9b97f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3b8d9eb3888f3aa305bd8c6324ac0099(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_184afe02b37963f5ed1514d749a33a9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b8d9eb3888f3aa305bd8c6324ac0099
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a9acbf6074e7c305eaa1e6c9d1a36b47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55395f03a5f450a07cd854814f459887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9acbf6074e7c305eaa1e6c9d1a36b47
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ba058ffcb25084ecc31a486c739658ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73ddeecb29b8489f683b4251557f7447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba058ffcb25084ecc31a486c739658ec
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4a6abb1384b1be101b8de7d38679892d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe2ca187c0c40de5b6e365c476fc2e20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a6abb1384b1be101b8de7d38679892d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55e70ae9d1d5524acfff38703759a94e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd01578e491d89190ede399f0322bdf5
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aecbdd166c5cd0b0653408c677302d7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21425b4fe1f82c9b8e07ae22c9db85a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aecbdd166c5cd0b0653408c677302d7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55395f03a5f450a07cd854814f459887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9acbf6074e7c305eaa1e6c9d1a36b47
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_973892f1e3a2a04d1d6b8ca9d08b02d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1152, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29fe933f32e58a950689946376c609ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_973892f1e3a2a04d1d6b8ca9d08b02d8
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fdea871e98687fb60cb8e471cfbfe1a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5de9c2601aaa73afa6783701686655d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdea871e98687fb60cb8e471cfbfe1a2
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2d6055eccc99fb2e5f4dfae77efbe3ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 26, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da32b8c4a220b31103e70342ab86bad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d6055eccc99fb2e5f4dfae77efbe3ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e8f6e1906ea4f6e0bc722112489d08b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 30, 50], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d3e7c432ab38fb1b375179d44b4b335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8f6e1906ea4f6e0bc722112489d08b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_528ed64cbcd73e3676107000bc1a1a75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b44657fa7fb4c58ebb119ae93d2d660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_528ed64cbcd73e3676107000bc1a1a75
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3026698925765b5a0df261c07531ba5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b340aafecb98dc83b7e91af5c324a6eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3026698925765b5a0df261c07531ba5c
    def get_inputs(self):
        return [
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_25e307d7b48611a530b0f38606c3cfc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 30, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d36e2169c3047f7b10be9603cb5f8b9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25e307d7b48611a530b0f38606c3cfc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e490abff7d45054751167f3af8623502(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 52, 52], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33897616e4bc151ae27b5db63b3c2e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e490abff7d45054751167f3af8623502
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7e20e17b9b8e7ee27df2fc8c34a37b9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e9e5ecb1bdec9b5e0d8bdafca987dd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e20e17b9b8e7ee27df2fc8c34a37b9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0f0176ba4d93a90b83842a8c06e2e82e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e5a2017aef075af327c395f866fb06e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f0176ba4d93a90b83842a8c06e2e82e
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3cd12a9e09307f567f54b307068dac17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 13, 13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aaaabe9a6e92a34b90c6374c81c774e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd12a9e09307f567f54b307068dac17
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6e059b671e56a2c87baa381eddd1a47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7b0ccdbad92e0b413f00da5640353cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e059b671e56a2c87baa381eddd1a47
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5e7b905892dfd8d8dd82876e606de55a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 34, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f99215f0e2a120c713ce8b9c6f3d865a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e7b905892dfd8d8dd82876e606de55a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_665941d3c0bf6ae5a5659d49378131d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49f1ab2dfe7e7bc2a65dbe925feeed68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_665941d3c0bf6ae5a5659d49378131d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_547e8a0a8b621ac1cd1a326a5b5b0f38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 76, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5cb64eeb061d1f3a8f5ca8976dae1f7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_547e8a0a8b621ac1cd1a326a5b5b0f38
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b6139ac6a6efd2e3f5c8bf0d510f39a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84a68563f1b9d896c0d75c4b1a663f28
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b853d11ccad613c90bf3765f6aa93d0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_618458b7a610ca4127004cde4de71f8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b853d11ccad613c90bf3765f6aa93d0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55e70ae9d1d5524acfff38703759a94e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd01578e491d89190ede399f0322bdf5
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_eff52a344d77e5c580fef83048425ede(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 36, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c0087f6f55cbf4ab2508b4dceffeb23e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eff52a344d77e5c580fef83048425ede
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_33b56337afff0eb234f134a8ea657e4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 36, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0193637402f774dc16a808a59f09f6c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33b56337afff0eb234f134a8ea657e4e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_abba74891cb443858bd07175f7cb0f5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 15, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9d75a7c6565f294f800c59323b34c04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abba74891cb443858bd07175f7cb0f5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21425b4fe1f82c9b8e07ae22c9db85a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aecbdd166c5cd0b0653408c677302d7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe2ca187c0c40de5b6e365c476fc2e20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a6abb1384b1be101b8de7d38679892d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aaaabe9a6e92a34b90c6374c81c774e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd12a9e09307f567f54b307068dac17
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ed26d424a2e2e7288381a3f510e4ffe8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3244fa7317c68bdd7323daf626849c4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed26d424a2e2e7288381a3f510e4ffe8
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e2684e87a08fa0fe3361257f6ea69bf2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 48, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c7367e121cb34a514b3e26dd0009fe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2684e87a08fa0fe3361257f6ea69bf2
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b8233c48289a7c4722a3e1e40a9c9d98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_705398fe6a8970db5946112544a501ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8233c48289a7c4722a3e1e40a9c9d98
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c74221f9c360cdaf143b11dd5ecd5508(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 60, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3b60f5ec6e321bb357976a310e0c89d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c74221f9c360cdaf143b11dd5ecd5508
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ed8b8d7712fb55ca032fc41689dec5ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d207c1bf67b292fd1cef7da1a79c721a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed8b8d7712fb55ca032fc41689dec5ee
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cc1749007a5e379f96e69f6104321d4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 36, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_796ed44f80c1cbe7ff214489d7a299a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc1749007a5e379f96e69f6104321d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_01ac7a76f9c7740d006c31b4e3b548e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_532bdc6f2d6ae42c76cf95a00c81934c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01ac7a76f9c7740d006c31b4e3b548e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3b60f5ec6e321bb357976a310e0c89d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c74221f9c360cdaf143b11dd5ecd5508
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2662bb5854e68788c2f48a443a4042e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6eaf0bafd4c8c10156a4c9ec881dfee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2662bb5854e68788c2f48a443a4042e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3e67089058f7f932304c0a9438bea37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_784739d2851e4e10c40cb833f72a65c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e77dc42c1ada96f0434ec8a21e9b51e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 21, 21], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cba7854684315e96887b9dbcc8212e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e77dc42c1ada96f0434ec8a21e9b51e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f99215f0e2a120c713ce8b9c6f3d865a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e7b905892dfd8d8dd82876e606de55a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1d14b1cf3d72d19958e7132d08a8714f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 34, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4376098c13dcf243d87c1691ce022466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d14b1cf3d72d19958e7132d08a8714f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8d23aaa41cbbbb8f42d2245c7e786e93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cfc7adb4876cffd89fbd4556a5f21d48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d23aaa41cbbbb8f42d2245c7e786e93
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e654f821433ef933d632b7dd5d7e36bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1ae6e585fb688c665f1c45034023254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e654f821433ef933d632b7dd5d7e36bf
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3fe2434160ae8798d78f2bf1ae01294c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 18, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2a8ea97deab56232da9519b0bc69b6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fe2434160ae8798d78f2bf1ae01294c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_63d72af4775c9464ee1f97342f99c91d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 18, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_636eaf1c399c09dc9802a202978c9fd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63d72af4775c9464ee1f97342f99c91d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8e7efc072f1785c1f2923359d2d57c44(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_685d868d521794209ec6f0956e2b3214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7efc072f1785c1f2923359d2d57c44
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3a68c9bf506d188183df93795a7ad50c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 17, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3575d7028816bd59022930108dc5670c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a68c9bf506d188183df93795a7ad50c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c6d785028c8b15f01d94a7c77800b938(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 17, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4d85b6dd7acf201e356fc9b1ea6b051(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6d785028c8b15f01d94a7c77800b938
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c0b8504f189ed7ad51038e643710a945(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d946a0dac1ce1b7c3ceb4d3a59666d66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0b8504f189ed7ad51038e643710a945
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55395f03a5f450a07cd854814f459887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9acbf6074e7c305eaa1e6c9d1a36b47
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_39cb1519cbd573c44d46925645faa843(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52d8cb84f1e1cf762df3cc9cee4c0c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39cb1519cbd573c44d46925645faa843
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe2ca187c0c40de5b6e365c476fc2e20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a6abb1384b1be101b8de7d38679892d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7d356abae6a2b1abd1a7e7c10f11bba6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ade1835a318f4fc4d560ab3f9d472c54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d356abae6a2b1abd1a7e7c10f11bba6
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2f22e7c6895ab9aa2b3a4e4e7861d503(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7908eaebeed6877f345eb127c93e200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f22e7c6895ab9aa2b3a4e4e7861d503
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bf478030a0beab11de50cfc551506eb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb4261d279392ad37790a5a00e27e26a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf478030a0beab11de50cfc551506eb7
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d207c1bf67b292fd1cef7da1a79c721a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed8b8d7712fb55ca032fc41689dec5ee
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21425b4fe1f82c9b8e07ae22c9db85a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aecbdd166c5cd0b0653408c677302d7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e5a2017aef075af327c395f866fb06e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f0176ba4d93a90b83842a8c06e2e82e
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0bbc4d330405812e0379613b566cba96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1152, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08a6fdb721d7e52f72ec77b28ec13ee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bbc4d330405812e0379613b566cba96
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_56a9dcdbf635d62dd50e83b13027edb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_344948ef5a10a045ea58a2e38a242cb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56a9dcdbf635d62dd50e83b13027edb5
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4680453c6af70e60800bc25e6b6dacbf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46d829d68a29bc8d792b843a8eae9acb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4680453c6af70e60800bc25e6b6dacbf
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3575d7028816bd59022930108dc5670c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a68c9bf506d188183df93795a7ad50c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dd65f7459ef453442506f31feb082a42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab97f146fb84821633e8b18e2a79d2b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd65f7459ef453442506f31feb082a42
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd987819c68956f588ce940932a8a263(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87b98a457fd1a39509bb0c8c855f4868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd987819c68956f588ce940932a8a263
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_027c65cfb1533afa36aacfc6564f7a68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d22d3af8d419e517729b109d9db3044(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_027c65cfb1533afa36aacfc6564f7a68
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3e03067fc6fa2402695b0d2e270b7472(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da2d57a3cc7090498f307c337389466a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e03067fc6fa2402695b0d2e270b7472
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0393ad6c001ba1d07b135e29193b8dd2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 46, 46], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_775706e61cad980b4641807400a3b012(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0393ad6c001ba1d07b135e29193b8dd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d61135ae7660da5144099c0a8a1123fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_727c5609e5d0df9fb92a92b41c67fbc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d61135ae7660da5144099c0a8a1123fb
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80023cb010b470833506c92e7baf741f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abe97bd705ac58eeb73fd18d9c007280
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb4261d279392ad37790a5a00e27e26a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf478030a0beab11de50cfc551506eb7
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d22d3af8d419e517729b109d9db3044(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_027c65cfb1533afa36aacfc6564f7a68
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7908eaebeed6877f345eb127c93e200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f22e7c6895ab9aa2b3a4e4e7861d503
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80023cb010b470833506c92e7baf741f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abe97bd705ac58eeb73fd18d9c007280
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a4d78cadc64bac2348ac9cbd5b12e2a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c5badc40bf316ddf51c5d87f1b6ef65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4d78cadc64bac2348ac9cbd5b12e2a8
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9268c193bdec87b2c55e0964baf9f8d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59d06cefa8c2ff89d17bd9c3dde45770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9268c193bdec87b2c55e0964baf9f8d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aeda1a2fdf06641ef14efea372799609(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47a3a41e728cc71ab1da30fa26de6868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aeda1a2fdf06641ef14efea372799609
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d22d3af8d419e517729b109d9db3044(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_027c65cfb1533afa36aacfc6564f7a68
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_55f64b096021a9d52c4e526a5dc66738(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 60, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74b8b311dc1eb3981badd93ed37a9fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55f64b096021a9d52c4e526a5dc66738
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1e282dabafaf4f4c9dbfbcae67342aa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af420d2e117539bb5208246cf4008bb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e282dabafaf4f4c9dbfbcae67342aa1
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_eeb32a25bbef504c8d387bb52b8df28e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_806898fc43dfe53f49d1b003302ed08a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeb32a25bbef504c8d387bb52b8df28e
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_788b3a81ea08e6b2b6eb68f9d5c9c1b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f73edf5fe68959b4feee2ffd7df780d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_788b3a81ea08e6b2b6eb68f9d5c9c1b9
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b602879dcc72fc9457890f3c13a46be8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_450aea18b5fa8a9cc5f4160a9ac16f44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b602879dcc72fc9457890f3c13a46be8
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ba1eaec9b3615501454fcf66741c8395(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79300f9455083bb8a3fe77e390e1c77c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba1eaec9b3615501454fcf66741c8395
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33897616e4bc151ae27b5db63b3c2e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e490abff7d45054751167f3af8623502
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9e22602597af58d71e11ceaa16756576(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bab5a14cd0af814036acf5d478a8df62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e22602597af58d71e11ceaa16756576
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_685d868d521794209ec6f0956e2b3214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7efc072f1785c1f2923359d2d57c44
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aba15f438f5cf0f75101d29c1d46eead(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7926cd0588e897ce63177ed890768f50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aba15f438f5cf0f75101d29c1d46eead
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80023cb010b470833506c92e7baf741f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abe97bd705ac58eeb73fd18d9c007280
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59d06cefa8c2ff89d17bd9c3dde45770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9268c193bdec87b2c55e0964baf9f8d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d6152c7f451b429eefaaec864145a9e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77358e96543f339905a71f67a288d191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6152c7f451b429eefaaec864145a9e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aaaabe9a6e92a34b90c6374c81c774e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd12a9e09307f567f54b307068dac17
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da32b8c4a220b31103e70342ab86bad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d6055eccc99fb2e5f4dfae77efbe3ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ab9567d1666e3855c0ac81dc7d69d4ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 23, 23], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f5833223e8da78d002be1ff78e57fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab9567d1666e3855c0ac81dc7d69d4ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe2ca187c0c40de5b6e365c476fc2e20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a6abb1384b1be101b8de7d38679892d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7b0ccdbad92e0b413f00da5640353cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e059b671e56a2c87baa381eddd1a47
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7b0ccdbad92e0b413f00da5640353cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e059b671e56a2c87baa381eddd1a47
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2b08d3a80f5159688d3bd42e660e297a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93621d96457f122444ab710cd11e0201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b08d3a80f5159688d3bd42e660e297a
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_532bdc6f2d6ae42c76cf95a00c81934c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01ac7a76f9c7740d006c31b4e3b548e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a8d2839102ccf8debe46fd9fa22a73fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66eff7718a3a636c3d7bd31c877abaac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8d2839102ccf8debe46fd9fa22a73fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4324327c5cb55691b2ec1e5ac6f223da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 19, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8ca85d70f042d4da05a5fc3f0aedb74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4324327c5cb55691b2ec1e5ac6f223da
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80023cb010b470833506c92e7baf741f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abe97bd705ac58eeb73fd18d9c007280
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_685d868d521794209ec6f0956e2b3214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7efc072f1785c1f2923359d2d57c44
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c7367e121cb34a514b3e26dd0009fe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2684e87a08fa0fe3361257f6ea69bf2
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b44657fa7fb4c58ebb119ae93d2d660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_528ed64cbcd73e3676107000bc1a1a75
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08a6fdb721d7e52f72ec77b28ec13ee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bbc4d330405812e0379613b566cba96
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e5a2017aef075af327c395f866fb06e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f0176ba4d93a90b83842a8c06e2e82e
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7908eaebeed6877f345eb127c93e200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f22e7c6895ab9aa2b3a4e4e7861d503
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_53e5ba2df2d6150777b7a0e1062f5353(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4490b678890a905ba3cc5d1ff0ddf6e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53e5ba2df2d6150777b7a0e1062f5353
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_14502fda1e129f2e7a51a268d0ca6167(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_830393159d1aa299bafd7bc1ec22c45e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14502fda1e129f2e7a51a268d0ca6167
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5c70c4d64ef46a0a6b4f6fcab8418134(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 15, 25], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0220200d73b7ed60bcfe9ac0a0104f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c70c4d64ef46a0a6b4f6fcab8418134
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2eeea89d6f03755a2f52d01d374d4eb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 96, 144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7fcf007ef181d1734ea9d4c90cb8622a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2eeea89d6f03755a2f52d01d374d4eb4
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f6fdeae9f9b20380a474c5ef4472d1b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 192, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49e0444552e1d310d0ab2b3eef9441d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fdeae9f9b20380a474c5ef4472d1b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da32b8c4a220b31103e70342ab86bad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d6055eccc99fb2e5f4dfae77efbe3ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0df282805c30eb71a5754cbee554e4bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 68, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb88adabfd0f9c18682ac924bfc1689a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0df282805c30eb71a5754cbee554e4bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d42922b49a8448a29d095a38edf5c538(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 68, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8dc59f58ca41cdfcbee30f93200eb82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d42922b49a8448a29d095a38edf5c538
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fcf007ef181d1734ea9d4c90cb8622a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2eeea89d6f03755a2f52d01d374d4eb4
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55e70ae9d1d5524acfff38703759a94e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd01578e491d89190ede399f0322bdf5
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55395f03a5f450a07cd854814f459887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9acbf6074e7c305eaa1e6c9d1a36b47
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d038a229a1ce2d1d58050b7dcc98e42f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f70f16dc16e18a711d7ebbde78faffd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038a229a1ce2d1d58050b7dcc98e42f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_89d4b8416eb58afb1a1ba42f2984e74c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1026a04d10be15265bbe75b4358d046(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89d4b8416eb58afb1a1ba42f2984e74c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_88194af946842c5db36a7b36f6e99261(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 72, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0119e8bf57a988a20aab612178636451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88194af946842c5db36a7b36f6e99261
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_12fac6db4d2e51abb791cea636de7a73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 72, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7be7a2d834026e1d4dee21c9a3f65ad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12fac6db4d2e51abb791cea636de7a73
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ac6e51b568a6131aa353b4fdee4d23b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 10, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32b5c8358dff754d8ba3f9491951adbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac6e51b568a6131aa353b4fdee4d23b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f70f16dc16e18a711d7ebbde78faffd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038a229a1ce2d1d58050b7dcc98e42f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2f4d92b7a08eba15703230af033f587b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2983fb59b8c638fb442e770d6fc8be10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f4d92b7a08eba15703230af033f587b
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87b98a457fd1a39509bb0c8c855f4868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd987819c68956f588ce940932a8a263
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a3a41e728cc71ab1da30fa26de6868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aeda1a2fdf06641ef14efea372799609
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b6139ac6a6efd2e3f5c8bf0d510f39a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84a68563f1b9d896c0d75c4b1a663f28
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70bd1dc0924d6b7eb6efca7fe92b144f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_446526674c1e45fbbe5dcaef803923c0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.3380227088928223]], [[1.423107385635376]], [[1.675360918045044]], [[0.7981780171394348]], [[1.6063164472579956]], [[1.4999457597732544]], [[0.9236422181129456]], [[1.7775229215621948]], [[1.698818325996399]], [[1.666978359222412]], [[1.6660394668579102]], [[2.412057638168335]], [[1.3759983777999878]], [[2.6064798831939697]], [[1.6997334957122803]], [[1.782984733581543]], [[0.759314775466919]], [[2.259439468383789]], [[1.8631319999694824]], [[2.137153387069702]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_1aa0a12f2813bafc0b06141626c762f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13203893eee5872df8e2a5c90635fb7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a6c3a2868e660a050e72bdf7f9c33c2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afc98df9019036dc717275780b3bd305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6c3a2868e660a050e72bdf7f9c33c2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_50005db0bcc73b7b06e2fa463ddbb52f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c848b2b48d0b47676a742a076cf105c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50005db0bcc73b7b06e2fa463ddbb52f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ab2d18be1f469f0a7d2d292bb76282bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 28, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_faf688820ea15011056b5d5b3fcc4612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab2d18be1f469f0a7d2d292bb76282bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fa01192323289b4139c78923532ab0b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1baa7f6064020e77c7a48c405c639075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa01192323289b4139c78923532ab0b4
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_53f0a615f17496420c155937757e74d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 120, 200], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0a272ad02cba2df7bce357ec2ed756d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53f0a615f17496420c155937757e74d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb4261d279392ad37790a5a00e27e26a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf478030a0beab11de50cfc551506eb7
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33897616e4bc151ae27b5db63b3c2e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e490abff7d45054751167f3af8623502
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7b0ccdbad92e0b413f00da5640353cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e059b671e56a2c87baa381eddd1a47
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d0a7f87e4c53ce0c1d89572e68684c48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89e328b7858392b532a3368c895978f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0a7f87e4c53ce0c1d89572e68684c48
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5ae3ed10c808785bf1106a3088733f51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 120, 200], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c22e685d683800c6ab4498924a6cf6b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ae3ed10c808785bf1106a3088733f51
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3244fa7317c68bdd7323daf626849c4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed26d424a2e2e7288381a3f510e4ffe8
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d71b3d2d4ef3dfc1e5914156d1c61997(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5b6af4c490e6a19d18d6faf2f9dc462(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d71b3d2d4ef3dfc1e5914156d1c61997
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9c38394f36b1c557852ed15552ed85d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 92, 92], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac00b191870dde5d1bddd37df2cc5d17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c38394f36b1c557852ed15552ed85d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_072f689b2145e5b8c8e62a5a0bb24d8c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_818c95fca381ae97c340fd07978c11dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_072f689b2145e5b8c8e62a5a0bb24d8c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_796ed44f80c1cbe7ff214489d7a299a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc1749007a5e379f96e69f6104321d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7b0ccdbad92e0b413f00da5640353cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6e059b671e56a2c87baa381eddd1a47
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de63d4b37d6e280bf7393f6132fa6467(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_446526674c1e45fbbe5dcaef803923c0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4385607242584229]], [[1.9105955362319946]], [[1.9766533374786377]], [[2.1473398208618164]], [[1.6893168687820435]], [[1.758873701095581]], [[2.361051559448242]], [[2.0224733352661133]], [[2.247314691543579]], [[1.9593055248260498]], [[2.2372918128967285]], [[1.4526458978652954]], [[1.9705458879470825]], [[2.8153584003448486]], [[0.9987118244171143]], [[1.834062933921814]], [[2.403637647628784]], [[2.5199103355407715]], [[2.047452211380005]], [[1.9269862174987793]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_1aa0a12f2813bafc0b06141626c762f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13203893eee5872df8e2a5c90635fb7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_afc98df9019036dc717275780b3bd305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6c3a2868e660a050e72bdf7f9c33c2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1baa7f6064020e77c7a48c405c639075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa01192323289b4139c78923532ab0b4
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73ddeecb29b8489f683b4251557f7447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba058ffcb25084ecc31a486c739658ec
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb4261d279392ad37790a5a00e27e26a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf478030a0beab11de50cfc551506eb7
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_685d868d521794209ec6f0956e2b3214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7efc072f1785c1f2923359d2d57c44
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87b98a457fd1a39509bb0c8c855f4868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd987819c68956f588ce940932a8a263
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe2ca187c0c40de5b6e365c476fc2e20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a6abb1384b1be101b8de7d38679892d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a53fec3a9a280fe58b32863cf5966393(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11564f0fdc015642a936e24201d99385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a53fec3a9a280fe58b32863cf5966393
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4aa6d67ba38eb3f0f3a6b0a7fb170ab5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8b446c9a0eea34bf7f3efba9852b18c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4aa6d67ba38eb3f0f3a6b0a7fb170ab5
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2b440e861500e0376a621f6a97237c0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_750f2492f8f2ee64743bd579d14ba954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b440e861500e0376a621f6a97237c0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab97f146fb84821633e8b18e2a79d2b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd65f7459ef453442506f31feb082a42
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d707d471bfb049328331c88f6c4d1eb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 56, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_059874e35d53568ad97b25d306fccfc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d707d471bfb049328331c88f6c4d1eb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_68f05256e574bf2508f93aa6c65881ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51c86c41803d22c93d59b8df221406d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68f05256e574bf2508f93aa6c65881ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e4386d091a548d19db1ca43401eb1df1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 42, 42], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_769743b9bb1eb9a634d9817ec3a2e67d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4386d091a548d19db1ca43401eb1df1
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_796ed44f80c1cbe7ff214489d7a299a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc1749007a5e379f96e69f6104321d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c38adb446670371897afc79b55f6a4d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc55b67402e84a635c8306603743bdf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c38adb446670371897afc79b55f6a4d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_855f8237a728824857e2cb22076d688d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2cefe84c5555594a255f3a1ce6a2a132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_855f8237a728824857e2cb22076d688d
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_978033477c29c6eb9b97c2b26b0cba6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5df6a29112132166357dda9f0d00081a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_978033477c29c6eb9b97c2b26b0cba6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d3e7c432ab38fb1b375179d44b4b335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8f6e1906ea4f6e0bc722112489d08b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bab5a14cd0af814036acf5d478a8df62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e22602597af58d71e11ceaa16756576
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_705398fe6a8970db5946112544a501ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8233c48289a7c4722a3e1e40a9c9d98
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_920b1c6f72bec78f663e28233ff1b4a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_203fd42deb597b936f25637af7d1cb78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_920b1c6f72bec78f663e28233ff1b4a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_344948ef5a10a045ea58a2e38a242cb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56a9dcdbf635d62dd50e83b13027edb5
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_193b0dc5eba13d4335033341c81b6189(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d795ccc70c7e48765fe9e9f4546db92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_193b0dc5eba13d4335033341c81b6189
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a3a41e728cc71ab1da30fa26de6868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aeda1a2fdf06641ef14efea372799609
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47a3a41e728cc71ab1da30fa26de6868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aeda1a2fdf06641ef14efea372799609
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c8608240d77aa0b08604cab7d218f6b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33f38355e8db41f08e260b9740a07825(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8608240d77aa0b08604cab7d218f6b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fd011c45ca280d8f8a4f1091a5ecd3b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af449241cacf26e1dafbf77db61e412a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd011c45ca280d8f8a4f1091a5ecd3b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_29c900106a3558ce1fc69a928e973a51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9, 14, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92b646d33c890cdaddfe14640735c066(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29c900106a3558ce1fc69a928e973a51
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb88adabfd0f9c18682ac924bfc1689a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0df282805c30eb71a5754cbee554e4bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7908eaebeed6877f345eb127c93e200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f22e7c6895ab9aa2b3a4e4e7861d503
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3e67089058f7f932304c0a9438bea37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_784739d2851e4e10c40cb833f72a65c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_eec5cc217f554257029ea7f2067e7c10(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1aaea529e51e1843f47d594d3ba55e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eec5cc217f554257029ea7f2067e7c10
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_617fdec5cd0b86c85b17013305ce2cb3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0bacbbe6feb1787fde986fcb40813188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_617fdec5cd0b86c85b17013305ce2cb3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29fe933f32e58a950689946376c609ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_973892f1e3a2a04d1d6b8ca9d08b02d8
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dc7ab02d2951deb425c623cfcef4b256(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 192, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c85ccea039ffa2613cadbd53ed9c7c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc7ab02d2951deb425c623cfcef4b256
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()