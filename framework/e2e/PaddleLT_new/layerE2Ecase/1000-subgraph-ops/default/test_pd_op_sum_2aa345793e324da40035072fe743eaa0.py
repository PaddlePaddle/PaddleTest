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


class PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_731f2e93b3d892a04dabeaf30b78d60e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([4456], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_1003dfd2e9ac6a919a7fbbe81a86a3dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_515cd7e4dec4213735cd4704b1325530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1003dfd2e9ac6a919a7fbbe81a86a3dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_6c61287b1b829ffa52d0131aba65652d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_721abf31f5fd4f730693d4658163b116(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_9b70b4a32baa450d67d24036897bdaae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23dd964ef3f4f0b232f3ddb20eee7beb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b70b4a32baa450d67d24036897bdaae
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_23dd964ef3f4f0b232f3ddb20eee7beb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b70b4a32baa450d67d24036897bdaae
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4736eb28890f5a323d41be8d7ab5e802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b70b4a32baa450d67d24036897bdaae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.031659774482250214, 0.011779126711189747]], [[0.0015809957403689623, 0.019061267375946045]], [[0.0002480687981005758, 0.14496245980262756]], [[0.002016015350818634, 0.001025476143695414]], [[0.13425667583942413, 0.01262865774333477]], [[0.06305055320262909, 0.0034878293517977]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d6abababfef4f9df7774a669731b287c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b70b4a32baa450d67d24036897bdaae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.04998870566487312, 0.03440047428011894]], [[0.016441909596323967, 0.09392662346363068]], [[0.0021563612390309572, 0.15451274812221527]], [[0.02709660306572914, 0.005870608147233725]], [[0.18218539655208588, 0.000573440978769213]], [[0.0007197426748462021, 0.0038797487504780293]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_c63bc54863f1fb60dca43bb6d174cc2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1696794331073761, 0.20723620057106018, 0.16646692156791687, 0.05010188743472099, 0.1094992533326149, 0.15132005512714386, 0.05072025954723358, 0.19371454417705536, 0.09878557175397873, 0.22637152671813965, 0.2711998522281647, 0.06406523287296295, 0.1390429437160492, 0.0888950303196907, 0.21650010347366333, 0.21968837082386017], dtype='float32').reshape([16]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5e4b8da2d275acbc69b328c3be0f4c1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_4625bf45ba0f560e2e4a2b73b1c1f188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_f97424bdb55962a3b21c703e392b1d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_088088e80f9eb20eb8c958f054db49e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bdc7d93745746acdc43dc1e4c57c512e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_bdc7d93745746acdc43dc1e4c57c512e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b8ef7710666c147be3eacd627f6df004(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5c4caa29a4a2ab033ef3cac011ebd798(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4918801784515381, 0.2867724299430847, 0.0616503544151783, 0.29163727164268494], [0.16717244684696198, 0.2702065706253052, 0.22598737478256226, 0.018749460577964783], [0.3956667482852936, 0.03867020085453987, 0.059548020362854004, 0.3594655990600586], [0.32695555686950684, 0.20531868934631348, 0.20880255103111267, 0.2319372296333313], [0.1041281595826149, 0.2930876910686493, 0.3555704355239868, 0.18245330452919006]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_bea92f254bf71e14f7bb14245d6b0b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2883334457874298, 0.2907085716724396, 0.23830997943878174, 0.09005986899137497], [0.0063690803945064545, 0.2998930811882019, 0.4675793945789337, 0.4290544390678406], [0.3478759825229645, 0.3107743263244629, 0.18506208062171936, 0.1849118173122406], [0.0063690803945064545, 0.2998930811882019, 0.4675793945789337, 0.4290544390678406], [0.3478759825229645, 0.3107743263244629, 0.18506208062171936, 0.1849118173122406]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_238687b94ccb92ad7bfcea0943e81ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1003dfd2e9ac6a919a7fbbe81a86a3dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_eda17b2ae69fee74b4734df9467801b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1ee9601301780427676bd9ca762eff0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_1ee9601301780427676bd9ca762eff0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b5b78ec07aaea5b15f26267c50a9d34a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1357516497373581, 0.40738099813461304, 0.44392189383506775, 0.09353798627853394], [0.03238692879676819, 0.22002668678760529, 0.4370597302913666, 0.05804285407066345], [0.22183875739574432, 0.4310569763183594, 0.3100452423095703, 0.13087481260299683], [0.03238692879676819, 0.22002668678760529, 0.4370597302913666, 0.05804285407066345], [0.22183875739574432, 0.4310569763183594, 0.3100452423095703, 0.13087481260299683], [0.20530103147029877, 0.26093873381614685, 0.25597333908081055, 0.21422967314720154], [0.20530103147029877, 0.26093873381614685, 0.25597333908081055, 0.21422967314720154]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_494c0a7a8647e8d1822acf5d9b1bf588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
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


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_2ca456cabe6af53626402872fabb978b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_088088e80f9eb20eb8c958f054db49e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0bc75020a42b438bfb3bb84c851cb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_e0bc75020a42b438bfb3bb84c851cb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b8ef7710666c147be3eacd627f6df004(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
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


class TestPrimitiveOp_3115968f0cbf352649737c7aa4abb4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17110438644886017, 0.21251949667930603, 0.12546300888061523, 0.26116546988487244, 0.06768501549959183, 0.10625585168600082, 0.020580770447850227, 0.16533957421779633, 0.09156273305416107, 0.004066269379109144, 0.05786304548382759, 0.18723663687705994, 0.20651261508464813, 0.04116661474108696, 0.13532692193984985, 0.17042267322540283, 0.09280594438314438, 0.25454822182655334, 0.18432599306106567, 0.2368009090423584, 0.16477525234222412, 0.019773436710238457, 0.22420437633991241, 0.015190720558166504], dtype='float32').reshape([24]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_c228125b50e761dc02c1da2ac102c240(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e25d9bf96e56a35d45868499336b8544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_e25d9bf96e56a35d45868499336b8544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ccd06a21a01deb043677e632eea5a15f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02784128487110138, 0.05515909567475319, 0.07934670150279999, 0.04918450862169266], dtype='float32').reshape([4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_93b85fde3cd239312faf587839e48da1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.31275293231010437, 0.14918148517608643, 0.39206165075302124, 0.28137341141700745], [0.0677771344780922, 0.31497669219970703, 0.3975636065006256, 0.2412356436252594], [0.005561307072639465, 0.018071070313453674, 0.16048210859298706, 0.16573911905288696], [0.041177887469530106, 0.02563190460205078, 0.13773715496063232, 0.11778953671455383], [0.041177887469530106, 0.02563190460205078, 0.13773715496063232, 0.11778953671455383], [0.005561307072639465, 0.018071070313453674, 0.16048210859298706, 0.16573911905288696]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_be96f59d3e9e63778bd967afe875e6e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22132810950279236, 0.2636357545852661, 0.0795370489358902, 0.10273897647857666], [0.1800602227449417, 0.3080964982509613, 0.04047614336013794, 0.03155049681663513], [0.11648650467395782, 0.05020785331726074, 0.03515303134918213, 0.061091840267181396], [0.01956409215927124, 0.46025997400283813, 0.06443309783935547, 0.19794350862503052], [0.22132810950279236, 0.2636357545852661, 0.0795370489358902, 0.10273897647857666]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_ef78903cbc83351f43b0e4b31a087496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
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


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_aed31eff26155394597c30c98803f36d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23681814968585968, 0.03353661298751831, 0.016777783632278442, 0.28043243288993835], [0.04496394097805023, 0.09608559310436249, 0.12442414462566376, 0.19400113821029663], [0.04946696758270264, 0.23147156834602356, 0.2051411271095276, 0.018683740869164467], [0.058955393731594086, 0.19874268770217896, 0.1565515100955963, 0.160660982131958]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_fe47771d12356178276d9022ed1b0c11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d258f4550a099f4d90025e44cdd2c27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe47771d12356178276d9022ed1b0c11
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


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_595cd43cb0a9de4d533b8bae5ae10960(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_93a354e0e89ff27f7808a1965d4dd5ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_0d7bcdaedd8fe851f3f0096dc35f7731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_25f7088a347e3f5d2cc51f89fdea6357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b8e32209f307c29c1a031740255e597d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b8e32209f307c29c1a031740255e597d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_fdb121395e5cb308cf7d1b3890cb3355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe47771d12356178276d9022ed1b0c11
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_95232b2c154e42f9949e0ab8b49d5258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05286979675292969, 0.135944664478302, 0.20292210578918457, 0.25536227226257324], [0.05286979675292969, 0.135944664478302, 0.20292210578918457, 0.25536227226257324], [0.1636086106300354, 0.02989518642425537, 0.12079188227653503, 0.2221795618534088], [0.18770182132720947, 0.15640157461166382, 0.0009250938892364502, 0.22462987899780273], [0.1361793726682663, 0.13610488176345825, 0.10240694880485535, 0.13521483540534973], [0.42806440591812134, 0.10874399542808533, 0.4567840099334717, 0.012656182050704956], [0.2613893747329712, 0.07425457239151001, 0.02400289475917816, 0.299114465713501]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_2c2aa15ec1330f30c8a2e98ff198bc53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1a6b9da94405dae9df233285e2325770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_1a6b9da94405dae9df233285e2325770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_1069d21e24014048f02bdaba9f4f75ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([4976], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_16c5740facf5a066d7cd6678707537ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([1176], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_bf247e3b3503a3869eee29e38a3ee612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1003dfd2e9ac6a919a7fbbe81a86a3dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5feb1483ca0478b42b4c3c6dec7eee8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6159609bc8b36d9a6e9ca50f753d3a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6159609bc8b36d9a6e9ca50f753d3a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_fe442d0e5342298946a2d150fd02c0bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13133330643177032, 0.08482658863067627, 0.2580347955226898, 0.09818729758262634], [0.10038453340530396, 0.046277258545160294, 0.16053301095962524, 0.016694992780685425], [0.10038453340530396, 0.046277258545160294, 0.16053301095962524, 0.016694992780685425], [0.0273551344871521, 0.002341151237487793, 0.11530663818120956, 0.42686018347740173], [0.03745202720165253, 0.1409778594970703, 0.37401413917541504, 0.4199698865413666], [0.12621213495731354, 0.11907048523426056, 0.40092790126800537, 0.29637330770492554]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_45d86e8c1b2b418d02aeb35af4f44852(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1aa599f0cf38ab49d12fa1bda4bf3ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45d86e8c1b2b418d02aeb35af4f44852
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


class PrimitiveOp_c58b2cafb3d155aae79f1d9783b871eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35459e59dbc7e04e7c22dde9822b7e38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c58b2cafb3d155aae79f1d9783b871eb
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7003aae419f01d5f6abdd4fb3e12c102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d1e38c0b89f853d68d9c9e21d049b98d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_d1e38c0b89f853d68d9c9e21d049b98d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_e45d909828368247fdacf7428368350e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5ad2207bf4fda7cb9b2bf9b2e56feabb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5ad2207bf4fda7cb9b2bf9b2e56feabb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_95fc28b8126ee79822deb196d6c3410a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_39f261ac35877b60a2f6e83b40540322(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_39f261ac35877b60a2f6e83b40540322(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_0844895ca01ce708accb13cbd75d4d42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
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


class TestPrimitiveOp_d02dd2a807f8e0126a5cf97150150178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13546261191368103, 0.23452575504779816, 0.23436672985553741, 0.045002181082963943, 0.011316009797155857, 0.03639400377869606, 0.011490323580801487, 0.13555288314819336, 0.2675095498561859, 0.09173566102981567, 0.13946524262428284, 0.14595440030097961, 0.1497451663017273, 0.1729821413755417, 0.20690588653087616, 0.1565496027469635, 0.22450672090053558, 0.07003117352724075, 0.22260484099388123, 0.1805296093225479], dtype='float32').reshape([20]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_2977d89883caa7fcf932b39e5645285b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([17406], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_50a225c6deb49bef879d59cc5cc3b05a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_e2d5103e7b5182bf284bf4d91c9a255b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_421a43d11ef9968350a1f723290c4d59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b12e37f10f4551c2d398a28d64de98d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b12e37f10f4551c2d398a28d64de98d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_dfe002d3d365cfab65d9cbcab282c419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b5fbb38b87ce535f5fc13eba68f19dc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.016613949090242386, 0.35582423210144043, 0.07691299915313721, 0.14768289029598236], [0.03833845257759094, 0.2901633381843567, 0.2641144394874573, 0.13769707083702087], [0.012129008769989014, 0.18745338916778564, 0.14885768294334412, 0.18658161163330078], [0.012129008769989014, 0.18745338916778564, 0.14885768294334412, 0.18658161163330078], [0.2790951132774353, 0.014318227767944336, 0.13249212503433228, 0.14775767922401428]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_26f9fc330814dec0b3e60f3793f2ee84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b92a772fc4df51599dbdc6be31d88a2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_60af401d7492c87e37e75fb1f6f6adc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_300e065cab84612627ac78249936a43f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dad93d9754d16737e24c3d919ae9cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_055770a7ec714744a85102ec2e915348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_055770a7ec714744a85102ec2e915348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0baac21a09c4c6369bf7ca2c59f8c83
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6640c295cec68d821576478c05f25f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b5d1e9e5f1ca9490df6208f7caa9112
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_d7af3d64f61e4751129e4708159afbe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1323268711566925, 0.23802849650382996, 0.3332147002220154, 0.11582101136445999], [0.013273239135742188, 0.00046723615378141403, 0.01889728009700775, 0.39437851309776306], [0.16595038771629333, 0.00343361496925354, 0.23501847684383392, 0.11366520822048187], [0.1323268711566925, 0.23802849650382996, 0.3332147002220154, 0.11582101136445999], [0.05888435244560242, 0.15700788795948029, 0.1641436368227005, 0.04239286482334137], [0.1870676726102829, 0.07733932137489319, 0.09762038290500641, 0.06075166165828705], [0.05888435244560242, 0.15700788795948029, 0.1641436368227005, 0.04239286482334137]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b4fbb6cf73d467898a6cca7cbc23f701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c61287b1b829ffa52d0131aba65652d
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