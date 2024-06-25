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


class PrimitiveOp_4240ed4b8d876e5930f293fb5a72ffc0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4344], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5fdb405dfc216e9eb0d584fd7d7ebd5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4240ed4b8d876e5930f293fb5a72ffc0
    def get_inputs(self):
        return [
            paddle.uniform([4344], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_95b0bd29a10988abb605e0e1cb54aa98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55f357e7306b0083e117d4bcf5a016f1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.00013748770288657397, 0.0001450754643883556]], [[0.03789886459708214, 0.032152775675058365]], [[0.0570400096476078, 0.02532128244638443]], [[0.001826585503295064, 0.03860104829072952]], [[0.007514291908591986, 0.07377305626869202]], [[0.0027487771585583687, 0.08789464831352234]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2de7bb9332a38cc7ceb0f1615ff5f975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55f357e7306b0083e117d4bcf5a016f1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.004213852807879448, 0.05482562631368637]], [[0.006549609825015068, 0.023551538586616516]], [[0.0018167238449677825, 0.02210627868771553]], [[0.10942615568637848, 0.03096318617463112]], [[0.00671613123267889, 0.06766187399625778]], [[0.02422056533396244, 0.032947611063718796]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_0b2986ca369124b44d4d5a7718a00865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c030f65a4c0c91e8ff4cf6af424d0d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.16035804152488708, 0.17226918041706085, 0.21067030727863312, 0.1782221645116806, 0.16634197533130646, 0.2240706980228424, 0.07091492414474487, 0.2300947606563568, 0.16219644248485565, 0.05300622433423996, 0.15612778067588806, 0.13565833866596222, 0.23084072768688202, 0.1890317052602768, 0.20128677785396576, 0.0678185224533081], dtype='float32').reshape([16]),
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


class PrimitiveOp_5d18f4f7364ec2eb19fe562254a12afc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_044d728e5840bb5f53a3e0e462d54fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d18f4f7364ec2eb19fe562254a12afc
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_044d728e5840bb5f53a3e0e462d54fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d18f4f7364ec2eb19fe562254a12afc
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_0233755de7aaea2aec34bda18cb85203(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa1b082232bd808630cfd1360de90d90
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.01077839732170105, 0.23139362037181854, 0.36757537722587585, 0.2106194794178009], [0.21976801753044128, 0.23060455918312073, 0.12467886507511139, 0.017615854740142822], [0.2861097455024719, 0.2822428345680237, 0.26744967699050903, 0.4451574385166168], [0.054540008306503296, 0.15870381891727448, 0.018163494765758514, 0.011783301830291748], [0.16896668076515198, 0.45631930232048035, 0.1181577667593956, 0.26534923911094666]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_b2c13e8ec83a11c43bd204c5ba02f8cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa1b082232bd808630cfd1360de90d90
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03768511861562729, 0.3141765892505646, 0.09385469555854797, 0.18678498268127441], [0.08251011371612549, 0.06584697961807251, 0.11500242352485657, 0.17597244679927826], [0.21548722684383392, 0.15471479296684265, 0.2636128067970276, 0.06203708052635193], [0.08251011371612549, 0.06584697961807251, 0.11500242352485657, 0.17597244679927826], [0.21548722684383392, 0.15471479296684265, 0.2636128067970276, 0.06203708052635193]], dtype='float32').reshape([5, 4]),
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


class PrimitiveOp_20c544d13bb11b0bc70d920c94356c08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c1b52a53907a617ff10d8c746cd0399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c544d13bb11b0bc70d920c94356c08
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6c1b52a53907a617ff10d8c746cd0399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c544d13bb11b0bc70d920c94356c08
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_4dc8f32bd33d4737f220e361f18532ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_371fc47f029152f4eedf5b6f68ea643d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05574280023574829, 0.19090011715888977, 0.07926230132579803, 0.08707726001739502], [0.07147979736328125, 0.2017761468887329, 0.04350680112838745, 0.36970213055610657], [0.30316054821014404, 0.08338502049446106, 0.24412307143211365, 0.23371408879756927], [0.07147979736328125, 0.2017761468887329, 0.04350680112838745, 0.36970213055610657], [0.30316054821014404, 0.08338502049446106, 0.24412307143211365, 0.23371408879756927], [0.09875491261482239, 0.18579769134521484, 0.22553396224975586, 0.21217313408851624], [0.09875491261482239, 0.18579769134521484, 0.22553396224975586, 0.21217313408851624]], dtype='float32').reshape([7, 4]),
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


class PrimitiveOp_0ed2618e4256e4d4f28cc373aa7b3719(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ecdcbf1cb497e2bdd4ae05a00ad9e781(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ed2618e4256e4d4f28cc373aa7b3719
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_ecdcbf1cb497e2bdd4ae05a00ad9e781(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ed2618e4256e4d4f28cc373aa7b3719
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e11586fc8a3477d1a49a6b164454e0de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9f3707e8a5cf849998d6eaad56e62bd
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07876385748386383, 0.14301396906375885, 0.203872412443161, 0.10648074001073837, 0.09389868378639221, 0.16273698210716248, 0.024190260097384453, 0.045463863760232925, 0.0815713182091713, 0.016841759905219078, 0.07825446128845215, 0.13594579696655273, 0.24945330619812012, 0.12225513905286789, 0.10318629443645477, 0.033306702971458435, 0.03634855896234512, 0.2128971517086029, 0.019355811178684235, 0.20264217257499695, 0.009467032738029957, 0.03820163756608963, 0.056723155081272125, 0.16698192059993744], dtype='float32').reshape([24]),
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


class PrimitiveOp_8ff84f6eeebb0caadc6e3d8a19149f65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89dd9ce9ad2709a2bb11aa4f5681fef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ff84f6eeebb0caadc6e3d8a19149f65
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_89dd9ce9ad2709a2bb11aa4f5681fef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ff84f6eeebb0caadc6e3d8a19149f65
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_31e6eab172752e8626a1f06528c98865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ef98373c8bcd3dbeb95e63d6d510d5f
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2645794153213501, 0.12126528471708298, 0.18659916520118713, 0.1863361895084381], dtype='float32').reshape([4]),
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


class TestPrimitiveOp_06685c0eb5a999f6cb21f349d6d1841d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97ee70c875c06f6230b360a65ca21e2a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19524270296096802, 0.11190202832221985, 0.24526070058345795, 0.04938575625419617], [0.2459895759820938, 0.23118141293525696, 0.38403549790382385, 0.043900057673454285], [0.04088515788316727, 0.16825607419013977, 0.2890157401561737, 0.1842956840991974], [0.08379620313644409, 0.3274226784706116, 0.012366145849227905, 0.11941033601760864], [0.08379620313644409, 0.3274226784706116, 0.012366145849227905, 0.11941033601760864], [0.04088515788316727, 0.16825607419013977, 0.2890157401561737, 0.1842956840991974]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_a93599bb4fcfe51792e8f728ef4cb519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa1b082232bd808630cfd1360de90d90
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.011681139469146729, 0.4303775131702423, 0.43140000104904175, 0.0924951583147049], [0.025084972381591797, 0.10988263785839081, 0.12713006138801575, 0.03839513659477234], [0.10030119121074677, 0.0020422935485839844, 0.024833127856254578, 0.0032770633697509766], [0.08440111577510834, 0.24669235944747925, 0.39913398027420044, 0.20100060105323792], [0.011681139469146729, 0.4303775131702423, 0.43140000104904175, 0.0924951583147049]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_d955708ef9dd3b62ec83868d11bcb022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8eb1a4144b9f7c5b1f17d27e072c97b
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14627555012702942, 0.13115960359573364, 0.14302843809127808, 0.10098734498023987], [0.23363037407398224, 0.13121597468852997, 0.03195944428443909, 0.3712884783744812], [0.20839068293571472, 0.00977557897567749, 0.09457176923751831, 0.17659586668014526], [0.14074213802814484, 0.15372057259082794, 0.018863890320062637, 0.19768190383911133]], dtype='float32').reshape([4, 4]),
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


class PrimitiveOp_cb61c89189691a362ffc57aa2268ea7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0743ea6e521b0211fa58eceed1eb752d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb61c89189691a362ffc57aa2268ea7a
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_0743ea6e521b0211fa58eceed1eb752d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb61c89189691a362ffc57aa2268ea7a
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f2fb3b13bb99511f91cad568f5e0f29f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_371fc47f029152f4eedf5b6f68ea643d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27607500553131104, 0.16805040836334229, 0.01612101122736931, 0.20125778019428253], [0.27607500553131104, 0.16805040836334229, 0.01612101122736931, 0.20125778019428253], [0.2777993977069855, 0.1656722128391266, 0.0767522007226944, 0.1466934084892273], [0.40827086567878723, 0.16954927146434784, 0.006458401679992676, 0.31475168466567993], [0.25782519578933716, 0.012537658214569092, 0.1426226645708084, 0.1456732600927353], [0.03390437364578247, 0.029349535703659058, 0.15627236664295197, 0.2299584001302719], [0.03064623475074768, 0.18180377781391144, 0.06553564965724945, 0.022069208323955536]], dtype='float32').reshape([7, 4]),
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


class PrimitiveOp_67b3b1af8b59e47a696d62f44ede16fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d96141ff28f55c5495fa66f278b3314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67b3b1af8b59e47a696d62f44ede16fe
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_0d96141ff28f55c5495fa66f278b3314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67b3b1af8b59e47a696d62f44ede16fe
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_15a8b39be3b532b332269d1d4f8a4610(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4807], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91964f5bea437ce4b49961c309ef38fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15a8b39be3b532b332269d1d4f8a4610
    def get_inputs(self):
        return [
            paddle.uniform([4807], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_f88e8cee4c956091a892206b1ff8885d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1205], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23d24754f80f08fb3376c436f6da9667(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f88e8cee4c956091a892206b1ff8885d
    def get_inputs(self):
        return [
            paddle.uniform([1205], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_19a732a02316aed732f8ac7016f9ee54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27102e4a4ef2318992fa65b2c0b5af21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19a732a02316aed732f8ac7016f9ee54
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_27102e4a4ef2318992fa65b2c0b5af21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19a732a02316aed732f8ac7016f9ee54
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_b7225e9ea714d316f47cc4d80280acd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97ee70c875c06f6230b360a65ca21e2a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.419389009475708, 0.2385486662387848, 0.17910200357437134, 0.25541141629219055], [0.15476015210151672, 0.12787313759326935, 0.274046927690506, 0.007582925260066986], [0.15476015210151672, 0.12787313759326935, 0.274046927690506, 0.007582925260066986], [0.19365759193897247, 0.06694699078798294, 0.38479316234588623, 0.05158597230911255], [0.36040982604026794, 0.0010609328746795654, 0.10762536525726318, 0.2891213893890381], [0.16834193468093872, 0.3301253020763397, 0.2282804548740387, 0.03077937290072441]], dtype='float32').reshape([6, 4]),
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


class PrimitiveOp_70aca6da0ab541ab6ce1f2208a880908(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d377d8a5c048843ab1c8ebc9eede77b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70aca6da0ab541ab6ce1f2208a880908
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_1d377d8a5c048843ab1c8ebc9eede77b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70aca6da0ab541ab6ce1f2208a880908
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_c587fbaf1d79fdaff4a4d3dd64b4711e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8254fc2b2e2e5ab7451826a32e97ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c587fbaf1d79fdaff4a4d3dd64b4711e
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b8254fc2b2e2e5ab7451826a32e97ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c587fbaf1d79fdaff4a4d3dd64b4711e
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_00518be410bf4bcf9d99351c4bc422e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03e0bb62f2e6a82ab0730fdc2e4ec638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00518be410bf4bcf9d99351c4bc422e5
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_03e0bb62f2e6a82ab0730fdc2e4ec638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00518be410bf4bcf9d99351c4bc422e5
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_793315a2a46df2090a0abe38b0e7740e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6de71e60cda215ef49a2271aadb10b45
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1934625208377838, 0.09184940904378891, 0.11118734627962112, 0.16306853294372559, 0.07168375700712204, 0.13675245642662048, 0.08526065200567245, 0.1264335960149765, 0.22446617484092712, 0.0634908452630043, 0.04249290004372597, 0.008992191404104233, 0.1914738565683365, 0.03277118131518364, 0.21949853003025055, 0.028292017057538033, 0.06627178192138672, 0.021751608699560165, 0.1513344645500183, 0.039255887269973755], dtype='float32').reshape([20]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_f4c4c5f1d392ad848e898517a8481cd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17627], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5bac048e96010fee0110eb82b95cdcfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4c4c5f1d392ad848e898517a8481cd6
    def get_inputs(self):
        return [
            paddle.uniform([17627], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_de3f27a3108f8676f8956a6e1e0cad98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_062146140829f9075ded2b72961e6718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de3f27a3108f8676f8956a6e1e0cad98
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_062146140829f9075ded2b72961e6718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de3f27a3108f8676f8956a6e1e0cad98
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c6784f3399eefc44ffbf0f573309ae6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa1b082232bd808630cfd1360de90d90
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05009490251541138, 0.15385662019252777, 0.2759556174278259, 0.3261295258998871], [0.1172669380903244, 0.12045417726039886, 0.021406829357147217, 0.13164407014846802], [0.02002665400505066, 0.08511902391910553, 0.2713954448699951, 0.060312606394290924], [0.02002665400505066, 0.08511902391910553, 0.2713954448699951, 0.060312606394290924], [0.3452134132385254, 0.04304805397987366, 0.06859937310218811, 0.2436056286096573]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_85d91ffad4beb36541a77ec9e52bba1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_371fc47f029152f4eedf5b6f68ea643d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10310405492782593, 0.2407267689704895, 0.1336056888103485, 0.3315361738204956], [0.4141213595867157, 0.22854921221733093, 0.3232783377170563, 0.018062174320220947], [0.1820223480463028, 0.009562350809574127, 0.15676060318946838, 0.008004188537597656], [0.10310405492782593, 0.2407267689704895, 0.1336056888103485, 0.3315361738204956], [0.37853386998176575, 0.035849153995513916, 0.40783774852752686, 0.12761946022510529], [0.1093188226222992, 0.21362468600273132, 0.153514102101326, 0.08548074960708618], [0.37853386998176575, 0.035849153995513916, 0.40783774852752686, 0.12761946022510529]], dtype='float32').reshape([7, 4]),
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