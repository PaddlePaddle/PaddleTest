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



class PrimitiveOp_9de2abc07ea04687f8d4ea76531a30bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 16, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9a47849c12fd9b9dd9aea67352c7016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9de2abc07ea04687f8d4ea76531a30bb
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6ff31ac08d740fe566438448e164db99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4347], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fb21b2e573358d61a5726b91566c52d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ff31ac08d740fe566438448e164db99
    def get_inputs(self):
        return [
            paddle.uniform([4347], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_717af2dacd56f8a9911784124147da0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8732, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ee8cd817f04e600d16f8ef5a8310c25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_717af2dacd56f8a9911784124147da0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_572dccf95314b07b7835634257d8fcbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf1a1d812bd5f525f264ece70edb9b8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_572dccf95314b07b7835634257d8fcbc
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_282a64d736cdb40b3e83f0118e7751bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 21824, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4377c9fea6945a6c582f6405daf20120(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_282a64d736cdb40b3e83f0118e7751bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4377c9fea6945a6c582f6405daf20120(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_282a64d736cdb40b3e83f0118e7751bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_55f357e7306b0083e117d4bcf5a016f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b4603e8fd71656780e28c3a99c4bd03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55f357e7306b0083e117d4bcf5a016f1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.03257160261273384, 0.0003440641157794744]], [[0.03474440425634384, 0.006207597907632589]], [[0.07631706446409225, 0.0007024301448836923]], [[0.07516123354434967, 0.14284366369247437]], [[0.022162185981869698, 0.05036007612943649]], [[0.0016247383318841457, 0.1023864895105362]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_88dca484aae3bc5c9f3fa193b484dff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55f357e7306b0083e117d4bcf5a016f1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.025917118415236473, 0.004457618109881878]], [[0.001760551007464528, 0.08213938772678375]], [[0.0262510497123003, 0.042987924069166183]], [[0.01287275180220604, 0.1913413256406784]], [[0.0007011961424723268, 0.07691141963005066]], [[0.06876079738140106, 0.0869438648223877]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_606b67899ccda1ebda4728cad9142ba7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 16, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc9b05fb3f7427b169dee68a0c16c84a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_606b67899ccda1ebda4728cad9142ba7
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a2c030f65a4c0c91e8ff4cf6af424d0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8120efac9f576be0948dcf1d9871ce69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c030f65a4c0c91e8ff4cf6af424d0d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11613192409276962, 0.20828785002231598, 0.25290247797966003, 0.13531818985939026, 0.02594972960650921, 0.14845767617225647, 0.004339678678661585, 0.2223743498325348, 0.12508325278759003, 0.028634779155254364, 0.026195762678980827, 0.053154170513153076, 0.04141264781355858, 0.00407301913946867, 0.2425984889268875, 0.0013169918674975634], dtype='float32').reshape([16]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_65676ea77eb2011096ca99160568ce4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[53, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf1a29c82c40c3fa43fe184ba8a64cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65676ea77eb2011096ca99160568ce4e
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_a7444ff0388870fc2c0c9a10f1e5446c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d44a65c2c86da6823ac8516fe34fb023(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7444ff0388870fc2c0c9a10f1e5446c
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_372950c09b4f0504a8255ea7e39c598a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5831b6ef77563a11f78812aeed341ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_372950c09b4f0504a8255ea7e39c598a
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_4acc38b71c13b497a2bf30f3ae0533ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2ce02b13d73621263602df9895264e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4acc38b71c13b497a2bf30f3ae0533ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_50854b00a7bc7d1bc082fcc12da61ef1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1786, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21859fec5be6d442464af0663a375db6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50854b00a7bc7d1bc082fcc12da61ef1
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_21859fec5be6d442464af0663a375db6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50854b00a7bc7d1bc082fcc12da61ef1
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_8cf28555477c7916449567a1c3759d2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4895dc9362b49eb1e093fd66735e8535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cf28555477c7916449567a1c3759d2f
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_fa1b082232bd808630cfd1360de90d90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2d451a483d150c11b70b923998c54b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa1b082232bd808630cfd1360de90d90
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23033301532268524, 0.14335858821868896, 0.2451336830854416, 0.01158994436264038], [0.1772065907716751, 0.3110963702201843, 0.2927146852016449, 0.011528611183166504], [0.24803833663463593, 0.11553388833999634, 0.05100834369659424, 0.2954212725162506], [0.02581791579723358, 0.06242261826992035, 0.31831613183021545, 0.10948888957500458], [0.1231008768081665, 0.16309943795204163, 0.2740582227706909, 0.014111340045928955]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_bfad3254a276ecf96b0f46b3513c09c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 4, 16, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5eb969235f3b160a5077d9a397972cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfad3254a276ecf96b0f46b3513c09c7
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_df9a2bf656758996c6a0743bb9b418a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa1b082232bd808630cfd1360de90d90
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15211257338523865, 0.13868305087089539, 0.07743817567825317, 0.17150861024856567], [0.10281427204608917, 0.09220492839813232, 0.14854612946510315, 0.10860636830329895], [0.09037956595420837, 0.021578580141067505, 0.001748381881043315, 0.06808262318372726], [0.10281427204608917, 0.09220492839813232, 0.14854612946510315, 0.10860636830329895], [0.09037956595420837, 0.021578580141067505, 0.001748381881043315, 0.06808262318372726]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_ceb183b2f3b7261e2b94818c5f0e97a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c0436da044c06c35ac4f9e7f825616a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ceb183b2f3b7261e2b94818c5f0e97a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_4e3ab821428e028ec0516c3ab477707c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1d02cdce80680a71996bf003b5e34ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e3ab821428e028ec0516c3ab477707c
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f595e1f619d68699cdc3e4ce14ebb1f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5529, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf22ab1a39a540ee0b99551ccc442b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f595e1f619d68699cdc3e4ce14ebb1f9
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_cf22ab1a39a540ee0b99551ccc442b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f595e1f619d68699cdc3e4ce14ebb1f9
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_371fc47f029152f4eedf5b6f68ea643d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_056e74b0363bf6de1e83c0cc2a77f25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_371fc47f029152f4eedf5b6f68ea643d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18005210161209106, 0.014196991920471191, 0.15634046494960785, 0.10224347561597824], [0.2896423935890198, 0.018510550260543823, 0.03333774209022522, 0.10551036894321442], [0.28359436988830566, 0.09202780574560165, 0.27413296699523926, 0.07617615163326263], [0.2896423935890198, 0.018510550260543823, 0.03333774209022522, 0.10551036894321442], [0.28359436988830566, 0.09202780574560165, 0.27413296699523926, 0.07617615163326263], [0.19862128794193268, 0.22186554968357086, 0.21049772202968597, 0.0943182110786438], [0.19862128794193268, 0.22186554968357086, 0.21049772202968597, 0.0943182110786438]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_fbc6c0fade7c3ffd3f70198bec6f7cff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb1eb16b0b5074f75dcf975fa78b7720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbc6c0fade7c3ffd3f70198bec6f7cff
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_d33e14080eed40f580ce5397f0207db3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 16, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d13aa7f4e134af19b8584350c38b3253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d33e14080eed40f580ce5397f0207db3
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_e01316141c553bc78448a072b8169312(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[103, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_acc8c8d46af80811098057310de778f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e01316141c553bc78448a072b8169312
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_d2ce02b13d73621263602df9895264e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4acc38b71c13b497a2bf30f3ae0533ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6bac18030b475651d04023073654b8f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1767, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b9a0a6ca3a4028de2e09403ab85039f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bac18030b475651d04023073654b8f4
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_4b9a0a6ca3a4028de2e09403ab85039f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bac18030b475651d04023073654b8f4
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_4895dc9362b49eb1e093fd66735e8535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cf28555477c7916449567a1c3759d2f
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_5f65a33111e2917ff974d1aebaa57cd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 16, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63e513d00418e1cf72bee2d5b6f6f8c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f65a33111e2917ff974d1aebaa57cd4
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d9f3707e8a5cf849998d6eaad56e62bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e96987bc877c217d3f214567306467c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9f3707e8a5cf849998d6eaad56e62bd
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23288670182228088, 0.1661081165075302, 0.18396571278572083, 0.17048513889312744, 0.07596195489168167, 0.24752989411354065, 0.0778462216258049, 0.15620556473731995, 0.13816016912460327, 0.12898734211921692, 0.2273525446653366, 0.07447738945484161, 0.10948345810174942, 0.07428238540887833, 0.23289775848388672, 0.1864568144083023, 0.24074016511440277, 0.18621639907360077, 0.09449658542871475, 0.1786903291940689, 0.04468932002782822, 0.05219544842839241, 0.24228961765766144, 0.01969080977141857], dtype='float32').reshape([24]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_e059a66bb792706ba59fbd6a54e5d09e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_499b0ba53af33994103259d81dac0316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e059a66bb792706ba59fbd6a54e5d09e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_98d179fdeaf7f86b3f24054eeb44f8b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c34ccc588cdbe5fe029828986fc3c6fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98d179fdeaf7f86b3f24054eeb44f8b8
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_c34ccc588cdbe5fe029828986fc3c6fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98d179fdeaf7f86b3f24054eeb44f8b8
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_305a15e60cee76800e43c8ac4005ba2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 16, 16, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b4d365d66800112f1da47e022cabcf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_305a15e60cee76800e43c8ac4005ba2f
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3ef98373c8bcd3dbeb95e63d6d510d5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_705938d85ae5d79bc1dadaff1ba80ef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ef98373c8bcd3dbeb95e63d6d510d5f
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03918982297182083, 0.01320456713438034, 0.03668104484677315, 0.23245161771774292], dtype='float32').reshape([4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_97ee70c875c06f6230b360a65ca21e2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5c89396c3bf3db47804fe83d3d33959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97ee70c875c06f6230b360a65ca21e2a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11206990480422974, 0.19980105757713318, 0.28472182154655457, 0.09120070934295654], [0.05969715118408203, 0.1464267075061798, 0.09999644756317139, 0.21400588750839233], [0.17537441849708557, 0.40648606419563293, 0.07399022579193115, 0.07907019555568695], [0.13181909918785095, 0.14245106279850006, 0.021578490734100342, 0.1497858166694641], [0.13181909918785095, 0.14245106279850006, 0.021578490734100342, 0.1497858166694641], [0.17537441849708557, 0.40648606419563293, 0.07399022579193115, 0.07907019555568695]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_df5ce0f721a6b68ee318b1e4c2002483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa1b082232bd808630cfd1360de90d90
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.014901027083396912, 0.004994511604309082, 0.00011786818504333496, 0.18285894393920898], [0.12429235130548477, 0.0870169848203659, 0.029167458415031433, 0.01861666887998581], [0.11092722415924072, 0.10363040119409561, 0.12077483534812927, 0.10269473493099213], [0.430207759141922, 0.124886155128479, 0.06110185384750366, 0.010581247508525848], [0.014901027083396912, 0.004994511604309082, 0.00011786818504333496, 0.18285894393920898]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_6104e918a828e744e0cd6a38a664c430(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef3aa11f64e2dad0ee28ed2a764fae0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6104e918a828e744e0cd6a38a664c430
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5b4d365d66800112f1da47e022cabcf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_305a15e60cee76800e43c8ac4005ba2f
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_b8eb1a4144b9f7c5b1f17d27e072c97b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_265bf6e33eb243e8c48a89cd998fa0ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8eb1a4144b9f7c5b1f17d27e072c97b
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3960532248020172, 0.3196747303009033, 0.031348731368780136, 0.14671191573143005], [0.43643319606781006, 0.027652651071548462, 0.11587986350059509, 0.09504431486129761], [0.05985875427722931, 0.3477330803871155, 0.21412833034992218, 0.27874112129211426], [0.266995906829834, 0.09098441898822784, 0.010822445154190063, 0.062081217765808105]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_b274c430473fc567910e4d2d9ef1ae84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 19, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8e740b0ea689cde51c707c04184f05a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b274c430473fc567910e4d2d9ef1ae84
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_63e513d00418e1cf72bee2d5b6f6f8c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f65a33111e2917ff974d1aebaa57cd4
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_ed83224c98affe31b8441e59aeb0d157(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[84, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9b26cd8013f4e61977ce092e4e342da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed83224c98affe31b8441e59aeb0d157
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_2f97abdd7eb586fa4084c35009f16e73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f24398e40d4571314c729b0611a571cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f97abdd7eb586fa4084c35009f16e73
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_17e3b2159c47d5597a5df1283ac3ae48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3fb08849d440186eaf096235e87798ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17e3b2159c47d5597a5df1283ac3ae48
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_ac107b4ca130806084a1924178ed8848(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4c3562551a48b31e97de15aaa706908(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac107b4ca130806084a1924178ed8848
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_05e3ff1c26d76c9d726f7002f1d4f264(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2010, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cdb4d7f48b362d3bb930ce26f1a595f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3ff1c26d76c9d726f7002f1d4f264
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_cdb4d7f48b362d3bb930ce26f1a595f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05e3ff1c26d76c9d726f7002f1d4f264
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_5fac78642898ad34087ca9931177b68e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 152, 272], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6783c1ea91724a726798864e08a84586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fac78642898ad34087ca9931177b68e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_c22dadf18170a7b17df9dc72934873e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_371fc47f029152f4eedf5b6f68ea643d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07020962238311768, 0.22973822057247162, 0.2077721655368805, 0.3234386444091797], [0.07020962238311768, 0.22973822057247162, 0.2077721655368805, 0.3234386444091797], [0.1312311887741089, 0.2076951563358307, 0.3393630385398865, 0.20015442371368408], [0.27245303988456726, 0.004318207502365112, 0.0672927275300026, 0.39573079347610474], [0.14881111681461334, 0.21468481421470642, 0.0474470853805542, 0.06096624210476875], [0.029213299974799156, 0.4640655219554901, 0.25611305236816406, 0.2912372946739197], [0.006981849670410156, 0.0655575841665268, 0.08795641362667084, 0.09838144481182098]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_e3eed6cae2804eaecfc4d49b4a5e806c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6ab4454eff12abce349b4cb445f00ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3eed6cae2804eaecfc4d49b4a5e806c
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_aff3ff4c67cd2ecff60e2107190a098e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4663, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b1f4afeeb166e0c4d2aee610c79cfd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aff3ff4c67cd2ecff60e2107190a098e
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9b1f4afeeb166e0c4d2aee610c79cfd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aff3ff4c67cd2ecff60e2107190a098e
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_ab20ec2514e27194084a6d325a44a332(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4836], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9caa4840df35d5a4a44a4c0352051069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab20ec2514e27194084a6d325a44a332
    def get_inputs(self):
        return [
            paddle.uniform([4836], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_1e85fee0880a7c62d04ac052242676fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1230], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_460ba6980963210f9c974da2f4fd3802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e85fee0880a7c62d04ac052242676fa
    def get_inputs(self):
        return [
            paddle.uniform([1230], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_817f39967ff57e70e77bb5d845d15f4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2434, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1800edbc3d6873d0e9d2377bf9a967db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_817f39967ff57e70e77bb5d845d15f4e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_61c951f2969c0e31ef53574fdb5275f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_72f601c45d1fe459df142b236724e747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61c951f2969c0e31ef53574fdb5275f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cdef1be1d48c00fa0f56e3a0699a765c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1090, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1b4ea45cccc2eb1a749c007d5271a23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdef1be1d48c00fa0f56e3a0699a765c
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_c1b4ea45cccc2eb1a749c007d5271a23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdef1be1d48c00fa0f56e3a0699a765c
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_88268538dcac6d03a9255f65fd3fda1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97ee70c875c06f6230b360a65ca21e2a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1750708818435669, 0.22437986731529236, 0.24328318238258362, 0.28154340386390686], [0.04457196593284607, 0.1831628531217575, 0.09731841087341309, 0.33579692244529724], [0.04457196593284607, 0.1831628531217575, 0.09731841087341309, 0.33579692244529724], [0.11138305813074112, 0.33443817496299744, 3.0517578125e-05, 0.1521768569946289], [0.0802762508392334, 0.2545437216758728, 0.1444905549287796, 0.30462419986724854], [0.15836916863918304, 0.22807666659355164, 0.14974167943000793, 0.13632053136825562]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_f6ab5f3b6cb6001b88b566e4c62c92da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 2, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ba92ff372c500a6be3a4ee638b26c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ab5f3b6cb6001b88b566e4c62c92da
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b90b58805aebd825544c299e098fb107(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 16, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a41c5b38c6d20fff0adf666e35368fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b90b58805aebd825544c299e098fb107
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bfe81724de8aa9e6a2db046d07098221(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 2, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9db8afed624a93dfdec2552732102bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfe81724de8aa9e6a2db046d07098221
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_be349f97e0bb6a202560e923b8821497(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbab5fd62bd2add6d285e5634b7aadcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be349f97e0bb6a202560e923b8821497
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c57b898346474df4389b64dd64ce81fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2374, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f3ee338123ef97a4d7efc5a24cea868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c57b898346474df4389b64dd64ce81fb
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9f3ee338123ef97a4d7efc5a24cea868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c57b898346474df4389b64dd64ce81fb
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_c8162be8d4edefc6d867852e5ff070fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_969e0dbb56d34f90007aa111a728911f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8162be8d4edefc6d867852e5ff070fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f51a87e92cb0a90bbd22b89f0a0fc0df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3058, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ec842ecc60f6c68251b598d7e0d7b3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f51a87e92cb0a90bbd22b89f0a0fc0df
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_2ec842ecc60f6c68251b598d7e0d7b3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f51a87e92cb0a90bbd22b89f0a0fc0df
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_3628fa8febb0edee25a6649af05aaf68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e8da5cf7d8952f21a1111beb4da2bae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3628fa8febb0edee25a6649af05aaf68
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ab89b62b93fa0df4f2eefe52c36e852c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3793, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5536e442616e1491288326bd814d24ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab89b62b93fa0df4f2eefe52c36e852c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5536e442616e1491288326bd814d24ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab89b62b93fa0df4f2eefe52c36e852c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_ca17f83042b5982d10f194a35c60b039(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e11f14dfec72ac15dfdc5cf13cf9f78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca17f83042b5982d10f194a35c60b039
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_aec1d01203b488795a31a726890a732d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 16, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6ce00536790a4ffe2cbdaf664b6aad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec1d01203b488795a31a726890a732d
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d1f9c2d45783ea313e805c926a8045ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 16, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f800f32d487218e923fb45a171de841e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9c2d45783ea313e805c926a8045ef
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d13aa7f4e134af19b8584350c38b3253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d33e14080eed40f580ce5397f0207db3
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6de71e60cda215ef49a2271aadb10b45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4df9a14565119a94ab3889a9131beec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6de71e60cda215ef49a2271aadb10b45
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03572522848844528, 0.21290557086467743, 0.11963855475187302, 0.166105255484581, 0.015084540471434593, 0.01913272589445114, 0.11554906517267227, 0.014296004548668861, 0.21046561002731323, 0.22956301271915436, 0.10707047581672668, 0.21076148748397827, 0.006590642966330051, 0.08765774965286255, 0.13353323936462402, 0.20612771809101105, 0.1601167768239975, 0.20003457367420197, 0.18344350159168243, 0.2153872549533844], dtype='float32').reshape([20]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_e6d4f77edd6ba0bfa5a8af15cea00a19(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17423], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_069434ac95ac7914c6eff1308e5f0003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6d4f77edd6ba0bfa5a8af15cea00a19
    def get_inputs(self):
        return [
            paddle.uniform([17423], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_e6ce00536790a4ffe2cbdaf664b6aad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec1d01203b488795a31a726890a732d
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_363dd05755690ca9bc0ad852a3bdc591(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d535eff0122a05f275c82ff08b08fac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_363dd05755690ca9bc0ad852a3bdc591
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_ef285f41a6f22613155cd1fbf10fb581(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[47, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a00919aba5115cc5c53b8f9aa37d10e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef285f41a6f22613155cd1fbf10fb581
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_f353004ab1080f7ea2671ccc66801e23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca6d1fb2ac7cd5b6314eb0e0c63a4de6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f353004ab1080f7ea2671ccc66801e23
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_72020c3b98974dd4a722d4e0d6222741(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2042, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4953c6e159774758744338c5b3e2b1c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72020c3b98974dd4a722d4e0d6222741
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_4953c6e159774758744338c5b3e2b1c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72020c3b98974dd4a722d4e0d6222741
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_9cf7a77033b0e1029928a64240b47fc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2, 16, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dae31eec578479a2acffc5d988d10a60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9cf7a77033b0e1029928a64240b47fc4
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_023df03e9cede4380f26b2f3b38b8995(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb911d5fd4b18926b7226ef653a3b6cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_023df03e9cede4380f26b2f3b38b8995
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_d328ef87f3e00e73c6d16f3a0574c4aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa1b082232bd808630cfd1360de90d90
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09222230315208435, 0.2660036087036133, 0.2386646866798401, 0.482435405254364], [0.2286505401134491, 0.020706892013549805, 0.09183747321367264, 0.08162347972393036], [0.3744547963142395, 0.11938200891017914, 0.17953604459762573, 0.15103879570960999], [0.3744547963142395, 0.11938200891017914, 0.17953604459762573, 0.15103879570960999], [0.020409435033798218, 0.09995437413454056, 0.31952378153800964, 0.4160843789577484]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_a41c5b38c6d20fff0adf666e35368fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b90b58805aebd825544c299e098fb107
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e715616309821a9826e74e906c5a8537(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_812dd5aeece9c9f6d1eb7f1eb291438b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e715616309821a9826e74e906c5a8537
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_999ec3857d71e0cb6af3cdde1be0ac8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_877170646c5458d776efa9242e776132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_999ec3857d71e0cb6af3cdde1be0ac8b
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_dbd6eb111010dfdc96cd1c6b3d8f0c4b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_567aa3afeabf30ed36cfb6381404be47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbd6eb111010dfdc96cd1c6b3d8f0c4b
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_ff551b42c3eabac04e1bb48b94449183(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_491f814a1ab1aa8738712b4654cdbb9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff551b42c3eabac04e1bb48b94449183
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e37b55ce9c7c7ce0a3391309a56c5a6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5bbd6a720fea84f4bde428bf74601416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e37b55ce9c7c7ce0a3391309a56c5a6b
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5bbd6a720fea84f4bde428bf74601416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e37b55ce9c7c7ce0a3391309a56c5a6b
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b92dd28736937c3bc7d323ead7a1e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c683f34723f8918a8847ed1c8ba69a9
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_1cb588c1369f628800a23afb06232cfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_371fc47f029152f4eedf5b6f68ea643d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.25657403469085693, 0.2911469638347626, 0.2680356800556183, 0.17256245017051697], [0.27216804027557373, 0.21796457469463348, 0.2322089970111847, 0.1816495954990387], [0.06819947063922882, 0.016032874584197998, 0.10692843794822693, 0.08240850269794464], [0.25657403469085693, 0.2911469638347626, 0.2680356800556183, 0.17256245017051697], [0.18578732013702393, 0.23836110532283783, 0.03987012803554535, 0.15123692154884338], [0.3416287899017334, 0.1117306500673294, 0.20233027637004852, 0.09810971468687057], [0.18578732013702393, 0.23836110532283783, 0.03987012803554535, 0.15123692154884338]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_4d81dd39a87ee6aa5e17df4d0286da0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[52, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3544ac6e8d5a261bbc7d6a1140d784a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d81dd39a87ee6aa5e17df4d0286da0e
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_f800f32d487218e923fb45a171de841e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9c2d45783ea313e805c926a8045ef
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()