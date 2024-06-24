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



class PrimitiveOp_c7e9868b5ab8ecf307dd1ab5e07d57a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65da5ada72485101f0a6f0c147e3fad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7e9868b5ab8ecf307dd1ab5e07d57a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3fcc0bbffe633349897b63850bb88c51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f1403a56c9f000cc1a1774db4285be07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_92cb719a58e611bb95c2eeb1f148340f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.85, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eeee9d285e0b0660b82b8efada989853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92cb719a58e611bb95c2eeb1f148340f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9138370752334595]]], [[[0.6244052648544312]]], [[[0.5716403722763062]]], [[[0.3775431215763092]]], [[[0.6712799072265625]]], [[[0.49873918294906616]]], [[[0.31190258264541626]]], [[[0.38474228978157043]]], [[[0.7390226125717163]]], [[[0.4968501031398773]]], [[[0.6428844928741455]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_83ed0973ddc5fb5ba7c8b22a2c44eb39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.1764700412750244], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e18c64c64a19beef4e5afcb80a70fb2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.875, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fae34cf4ca5a15692e4e8b61139601de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18c64c64a19beef4e5afcb80a70fb2b
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a6183404a7ecc26c2a712a8aff68376b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.1428600549697876], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c779e0b4fc1d2cc7cd3d73d4c88aa72a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.3841710090637207]], [[1.0808602571487427]], [[1.3273292779922485]], [[1.3757654428482056]], [[1.539886474609375]], [[1.1660341024398804]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_12e1a2b78d530b512b7d05939b5c4a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.2320888042449951]], [[1.2548106908798218]], [[1.4815587997436523]], [[1.5624945163726807]], [[1.0106667280197144]], [[1.2702006101608276]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c998d4dce0aee6603b3d084197c94544(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf87c6ee7b764e206bbe248301c76d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74b12efbbae4ec5cf8ebc999d43e76c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d84398f290e2e18be8818fe576ad9d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e817bfd92546c062f99d6e3849de96e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_43541977669dddc3df1669204fd03f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a4c0e0f840297ae1d00580d3646da73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_43541977669dddc3df1669204fd03f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4a4c0e0f840297ae1d00580d3646da73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50bf9889e4aac2a0c803cd663dd0d1f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7780fc37244c219e8d1729848dcf68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50bf9889e4aac2a0c803cd663dd0d1f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7780fc37244c219e8d1729848dcf68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d4a5fbeb2c44550536f18aa171ea948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_608d05873d335b7ce8e7fa70a75b1590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d4a5fbeb2c44550536f18aa171ea948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_608d05873d335b7ce8e7fa70a75b1590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a3431ecb6c218dec0cca1e5e3272f146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.05675641819834709]], [[0.42963168025016785]], [[0.36815306544303894]], [[0.04792821779847145]], [[0.4340068995952606]], [[0.1019599512219429]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_169a55619c04bd86eaa1843a211ea11a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3304600715637207]], [[0.3483187258243561]], [[0.2565034329891205]], [[0.006738851312547922]], [[0.4289870262145996]], [[0.42194366455078125]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e9a2e205fad6418f3d71854ba46b1747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.06396379321813583]], [[0.27696508169174194]], [[0.21405671536922455]], [[0.360918790102005]], [[0.4037555456161499]], [[0.4615171551704407]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3238c9deb43c5b617d17a0d106026715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.41534051299095154]], [[0.3411482870578766]], [[0.18817031383514404]], [[0.3956049680709839]], [[0.32522040605545044]], [[0.21647119522094727]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_105f69babddceb717ec17b4fd33af5ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6dead01985697ad70453431238246289(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_46f4699da7c0237c8449b4859ecbd10a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b5c26768b2e651ce5bf6f0ea6b08912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f4699da7c0237c8449b4859ecbd10a
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_60a773efb857390366a1817c7c11ced1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ecb560248c45de8d07bbdea22761fd57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_42cbe7d21769457608268950030ac570(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ea4fd42fc6e46144cee2dc287dc5108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.to_tensor([[1]], dtype='int64').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_007487208e4091dcd686a62928d8f388(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa887bc8ad0203393fd9eaa921378ace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24010474979877472]]], dtype='float32').reshape([1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9246d467a3e274084a89f947fb2d5d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_882ce11f90604ed108803d9467f9d12d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_903db3f0f1852545bc40bd94f1c829ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f4699da7c0237c8449b4859ecbd10a
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0748f76ae71e657616493d2c23fb0745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_00e58f0b41d8d808c7ac38336f4571d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b43a3159ed1f25ecac9467e2a3b399cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5e0d753987b7ed054e827d1837e2a2da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dc56dcd21859e5e82d23653ead60d888(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4a33329d53e368421bc2f6dce214e237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d00f560e856c3016c02bfc970ab226ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_31ba3452945f5305f3d610d19c405e0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_00e58f0b41d8d808c7ac38336f4571d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8604246ca2a12d8d6f45997d7178bf73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5e0d753987b7ed054e827d1837e2a2da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_482f8d532c88c16a95276c62fe73783f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d42d98ec788c2939679a36e9d01c9d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_593464b9b0a2c1b0d759076eb9ce067e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_aec9df8215b8632e8426a1cf510bec35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_927bd12c47d22a6802113e46843955df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c18db5c7e5035525526edd8119315954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c7d1581b83043a11b0ab26db69451db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e43bd7c01600d27ffcf6d8139c18f544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ee1bb5a9199f10cf5e61f331fbe67a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(1116.034912109375, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9fa7bb9a041e0fec8f8e745760353eb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(166.53453063964844, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f5ea4eaf2946d4bf2028dde7ac18e414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(5.585515975952148, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab7604ede3c9ebfa556a276ed9cf6c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0007042166544124484], [0.0027819345705211163], [0.00519100297242403], [1.329875158262439e-05], [0.008006745018064976], [0.029768509790301323]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a6ad5530bf47da13bca514fe62cf753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.00017917051445692778], [0.001508450834080577], [0.001031440100632608], [0.004980630706995726], [0.022327451035380363], [0.0007504608365707099]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a27fdc4ce85c88caaef164a816ed69f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-6.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5cfbbac49930c1a4447266734277e0cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.03387898951768875], [0.08556250482797623], [0.07029067724943161], [0.20596033334732056], [0.11955881863832474], [0.1918669044971466]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([0.08333329856395721], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5be673ee59408e724aad21898b2f077d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.002823248039931059], [0.0071302056312561035], [0.005857553798705339], [0.0171633530408144], [0.009963231161236763], [0.015988901257514954]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([6.28318977355957], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1be8d539a11e926c7ef4d41a41850f3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.017739003524184227], [0.04480043426156044], [0.03680412098765373], [0.10784060508012772], [0.06260087341070175], [0.10046130418777466]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6cfe66c26bdac34f0760014ac45b1e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_301abcea76b31248bdbaa98571763af8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d625596fbb90a774d7223023cb10b731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e66d92eb1d14b7649473890dcad8bf4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.9125, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_789ce5960cff81d2ca84d30f7aac4562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e66d92eb1d14b7649473890dcad8bf4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7e93681d09d50e72e6f3ae38272e13c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0958900451660156], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_55a3bda13b7b5451c17eef816c49b983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_079b021238f4ac5824a617c7f23bc5cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55cc184a027da29ca4863d00f67a9d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_079b021238f4ac5824a617c7f23bc5cd
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_67f2a325da04ef601d88955f14a034c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([2.188481092453003, 2.0165317058563232, 2.16790771484375, 2.059605598449707, 2.0496935844421387, 2.1605939865112305, 2.0899791717529297, 2.0973901748657227, 1.975513219833374, 2.057857036590576, 2.253582715988159, 2.0845351219177246, 2.0382754802703857, 2.1864006519317627, 1.899365782737732, 2.0672075748443604], dtype='float32').reshape([16]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9bfe57ef45cbe9660f1eed42f6508e68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(1.9030299186706543, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4b07e4f32cd2e941051ce6b238a2b371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c02e6b60e330d193e25c58ad7b16dbed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_34070266870eb07f689ca4b57447ac79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_93bbb5012ffd792cab9274589b26dadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_93bbb5012ffd792cab9274589b26dadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dcd6a9055cab2aea1d839f66ff389e32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cf87c6ee7b764e206bbe248301c76d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_16c7cf0ed8590d19b4f35925b4c042d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9f92224ed0d677bfe18ca37ad25d64d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5d66d8bb827c590080899c3c8ed3b024(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a75dc011e6a8ac0d9fe52f11c235033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a842480527091eaa7171699b06709eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(34.54819107055664, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e18464205110e45f6ba5a29177799637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d3d02151dd6a9e52c32da5569d862e2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a2d707cb6f76dcaeb13820794a17f2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_13a502ca63ba0e9b6c90ae1584e97dbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6b7b5a22e95e7da56db9aa30eb6527f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cd61a397e033fb21143823daebfc46df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_63e314cc59ca2cee2eb25ea65b618bfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab87386f65602612915d8fafdcd905e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3edd0e6905b2afdc797089f8f0760503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_294ced270fa733b0c00dcd38cccb4d45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_294ced270fa733b0c00dcd38cccb4d45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_025b06bfc4b1aca281c583657ea9ebd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0cb273c101dfc41358431265f595f75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bec48de7d8073d09ef81446122d720d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57280b62014d0af5f75e25b7626814ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1745, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_131b247d9b8a06f6a086ccce1860ce7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f3178a5f3dea30575f7db6a58f7d0254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1745, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f3178a5f3dea30575f7db6a58f7d0254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1745, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_00eab247826505547715ad9fe9b5a46c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e31370f62c9ebceb73602d27dafa0e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f4e09da6d92e3a218814a10a67031a9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18c64c64a19beef4e5afcb80a70fb2b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5450965166091919]]], [[[0.3464287221431732]]], [[[0.0712399035692215]]], [[[0.7746424078941345]]], [[[0.6635845899581909]]], [[[0.47114866971969604]]], [[[0.9392749071121216]]], [[[0.4967826008796692]]], [[[0.6132677793502808]]], [[[0.09783613681793213]]], [[[0.32791659235954285]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b5c78bae1eeabc3d4f1aa50d01601dfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.1428600549697876], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1bd00524ac898769d2f7398c510382e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.95, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38660094e1a7046c78a3ac93406f1ea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bd00524ac898769d2f7398c510382e4
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5495796203613281]]], [[[0.5121719837188721]]], [[[0.5342656373977661]]], [[[0.16940586268901825]]], [[[0.04960491508245468]]], [[[0.7272676825523376]]], [[[0.25803056359291077]]], [[[0.1558159738779068]]], [[[0.3155585825443268]]], [[[0.5911800265312195]]], [[[0.42283105850219727]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7ee57939c78e38e3daa9264c0893d3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0526299476623535], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bd5c6dd151e11db9de2871031def904f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_215513fbbd659c5bf0445c85ff7b760c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1f408225c94353d9a9492186076bef96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(75.87244415283203, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04a5080159b0e38412fa2ba241c969cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(3.763566493988037, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4f43552da2d2808abc5c274d39cad031(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_15b4f95a6bd81495bbc71439af94abfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_84bf68f74480f4c292f9debb832704b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fcbbeba6137e38c90c2b5c6edac27445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1cb629405fc3e1f9a7b39dda62cae4c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ad264a53104a690b0df0fbb867a5ea5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_513ecbbb9060dbd3f7f09c89e51e79da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e8c5dde185a76f8744e446243e38df79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_720047a284ff1b5e950e51949d78a079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_81538981c2e04bc234e28dd4efac4af8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_299e96a406e7aba4bcb3f12b8b9852a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9debec52f129261395f31a17a5190c99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50bf9889e4aac2a0c803cd663dd0d1f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_513ecbbb9060dbd3f7f09c89e51e79da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_43541977669dddc3df1669204fd03f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_720047a284ff1b5e950e51949d78a079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3a7bff229d2d7775525f81c70d888037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f2877c897a46bec5ba145a0efa498aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_682b46477ff12c598f5940f4a184f0e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10f43a4157f8f2e9df3bec98171f0e06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2daa04075c235ce8916aea7a0e2eff53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(164.6451416015625, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0725de7432752907cbbd155d1df06f93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(3.273749589920044, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_15e699dbcea2e3a941b73d9135c26a1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.975, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b4552a1055bdc9a01afef0dc0e7bd6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e699dbcea2e3a941b73d9135c26a1f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1cc2f82d37f24d68670cc398c5d1a507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0256400108337402], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d84398f290e2e18be8818fe576ad9d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5292001fa4205f8fe851b87cfb66ab22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da2ec15c4d15f71319207b2ef9074b05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aaaf7730cffa8a3dac82578618ca2ad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a762bfdf433653ef4e5581040e698ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7a32448a99f886f445de701586237491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_775e2e0537d868253851cd5c4cee62c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.01010000705719], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_908986754c2db98b405d8d1b846a5fb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.08137653023004532], [-0.05491970106959343], [-0.010116374120116234], [-0.031606074422597885], [0.11603888869285583], [-0.03504899889230728], [0.028307661414146423], [0.18799147009849548], [-0.011176660656929016]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5639e38f3c0441ee8c0a297ae2df5859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.042215973138809204], [0.03768906369805336], [0.03922963887453079], [-0.001653090352192521], [0.12294206023216248], [0.12214275449514389], [0.05118217319250107], [0.18799147009849548], [0.01930209994316101]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97456327be7dbf67ff2b0233f5155550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.9276241660118103], [-2.4571785926818848], [-1.2578758001327515], [18.119388580322266], [0.2672225832939148], [-1.2869510650634766], [-0.4469234049320221], [0.04201709106564522], [-1.5790386199951172]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e3af2274b881c98326ecbad7b227a723(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0723758339881897], [3.4571785926818848], [2.257875919342041], [-17.119388580322266], [0.7327774167060852], [2.2869510650634766], [1.4469233751296997], [0.9579828977584839], [2.579038619995117]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58115c3034db778bfd0b37040abe17c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_255bf29b870dd309bbebd507df2f4fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(11684.9267578125, dtype='float32').reshape([]),
            paddle.to_tensor([0.09090910106897354], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f2106da3593c4a0f49a399956f30bbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(1062.2662353515625, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8590abc1be2fea1a0ea56e8cff905e19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.036700814962387085], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_26f5c28b0faae2d5df720296645542e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.4094476699829102]], [[1.194676399230957]], [[1.0145188570022583]], [[1.558000922203064]], [[1.5964288711547852]], [[1.2101774215698242]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0131449609d4146473d1b08157cf45ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.0817480087280273]], [[1.5037397146224976]], [[1.0871819257736206]], [[1.1630326509475708]], [[1.6129393577575684]], [[1.0788321495056152]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f58c1e442ee9324d2a06b875a0166083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_adc5dc1a9beaedf06881268b87297646(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c09455eee5362f2a8a0f5519a808c7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc5dc1a9beaedf06881268b87297646
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f58c1e442ee9324d2a06b875a0166083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2573fc03959dad1eb3f9b560dabb23b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bbab847187bde0d5119ae0e3e48479cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bbab847187bde0d5119ae0e3e48479cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9ed194a7124afe14ce2fdc2d232536c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_143af4757bd1f64b0acf7c5ea765b853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_71100401709368c6cf46249c20654e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5556, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_199c0a36957b72561eadbb619e9ce908(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3ba8c06ee07ec60a364291588ebcc935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5556, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3ba8c06ee07ec60a364291588ebcc935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5556, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d3dc7b27b52743c4063fe9848612ef20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8d9ceeec955434cac90ffdf4a46b08f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([86, 1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d92318522d05d37a2b19b4fab0b44e56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([54, 1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_06ff4047dd00b36841984b859d27834c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.8375, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9bf0bd13291a311de4c58635a4fb24d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06ff4047dd00b36841984b859d27834c
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_15867f09a12d42edc32718b027524d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.1940300464630127], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_878fc9384341c9854f3b31dfe3f804a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(145.67063903808594, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_73e89567c94abc9c299769bf2fc50a5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(3.2351479530334473, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_833592f829768195ebd80092343e192d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_079b021238f4ac5824a617c7f23bc5cd
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_81642496ae5b608b6511328d32f99ac0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d7a0fa1da7cfa9bb2a7350049064b98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(5.719027996063232, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_51e3741d9ded48630510e131e55a134a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a9d7d95d3e71883966ca1a61b0b63baf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4762192368507385], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_35ad818b8f3bad5a26d51c21d4f9ad41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06927169859409332], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5b4f9133bb72b33f43a1784cb0ab8977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([1.2122366428375244], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_82fd7bceac6ca1750cb2cfb9fcfe747f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4d82c8364adcc3c639c5e470caed7828(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d800197d27b942ebf684797bc2a79cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f4699da7c0237c8449b4859ecbd10a
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c9f16ded2aa1196f458e6de4f180f1d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(135.54122924804688, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_495ff257a071bf4506c7c4c2bd351200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(71.04558563232422, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_010a0314d3593cce98c2a494ec7bb590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_775e2e0537d868253851cd5c4cee62c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.01010000705719], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_75971479e8081283c1a75a4e3e794e2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11827237159013748, 0.7674310207366943, 0.4971865713596344, 0.3340417742729187, 0.5272749662399292, 0.6047372817993164], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_98782f2e41579f7d26b78d3fdcbae2dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.560920238494873, 0.4693235456943512, 0.6441035270690918, 0.5429595708847046, 0.7865118980407715, 0.23787955939769745], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d32938a2876fded8a7c88322e158a694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15977796912193298, 0.7099542617797852, 0.16918303072452545, 0.39283862709999084, 0.5733265280723572, 0.6824370622634888], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f31f0971b8d494d03b5c9b725ee74df6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34862324595451355, 0.08672405779361725, 0.5035572648048401, 0.4582551121711731, 0.7306879162788391, 0.01927357353270054], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7f6e893b1e74b938d469fd1956bd5fe0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c80756be2ffd3f1dd174d160aaff1142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6e893b1e74b938d469fd1956bd5fe0
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.00635618856176734, 0.0034854901023209095, 0.019583148881793022, -0.03482642397284508, -0.03631962463259697, 0.0001548959407955408], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d99bfc4c53c4a9ff44b0c43cb74dcb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6e893b1e74b938d469fd1956bd5fe0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.011698182672262192, 0.03742148354649544, 0.03183488920331001, 0.0026579787954688072, 0.001309265848249197, 0.01345645822584629], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53fe80018e319fc2517fe2013142793d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6e893b1e74b938d469fd1956bd5fe0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.005829278379678726, 0.17400214076042175, 0.2847479581832886, 0.12188687920570374, 0.01871480792760849, 0.13883072137832642], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1564038880607d89d3f7240502f5cd79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31478211283683777, -1.4646350145339966, 1.729736328125, -0.9953817129135132, -1.7009549140930176, 3.011219024658203], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4052850008010864], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_33fc97b08736fb3dd9b6dba9d150c721(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24c842fc8b2d947b938aa3d5fee17602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33fc97b08736fb3dd9b6dba9d150c721
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, -0.0, -0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1822827b55b4beb0648c5df2a60d1a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6e893b1e74b938d469fd1956bd5fe0
    def get_inputs(self):
        return [
            paddle.to_tensor([1.040158748626709, 1.8693994283676147, 2.2126078605651855, 1.4015501737594604, 2.1725897789001465, 4.674897193908691], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_25d47aac6c0fed0f82443ac1f454869f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([3.008348226547241, 1.6193939447402954, 1.7763633728027344, 1.1368528604507446, 1.7028288841247559, 3.9857332706451416], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8acc7ee69c4da784609602a7e8b2969d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(2.2049200534820557, dtype='float32').reshape([]),
            paddle.to_tensor([10.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3fcc0bbffe633349897b63850bb88c51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_92d9a6dc22f823af185291fdba812995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_92d9a6dc22f823af185291fdba812995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_25d78eee677c93b6f0c25447645a4fb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ad6ca1eb7fce81b83fe14851c1f4cc7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d2ad3b87e1048a7a6aaa35e2ed9dedc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1744, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_61dc20dbbdeea77553db8838e678f9bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_96a0c65442ab89e3c53601857861ab7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_323ba2f95b06d4540ae9ccbeb44c4af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96a0c65442ab89e3c53601857861ab7d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1744, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_323ba2f95b06d4540ae9ccbeb44c4af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96a0c65442ab89e3c53601857861ab7d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1744, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59352ba0e82c2ea747ee64c93335ed78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ccef0a97dbfbc00a31f1382d12be7ea2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c08f75bc7f7f6a9cfca90f2b9ed56af6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3aa9a1906cbb9d7960b6d6014d185e9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b41e0aa8a46b2279e12188462c36198a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3aa9a1906cbb9d7960b6d6014d185e9d
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d400b442b7932955d734fcd6d46844ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.4153806269168854]], [[0.19716790318489075]], [[0.23610685765743256]], [[0.45234987139701843]], [[0.10462033003568649]], [[0.38143959641456604]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_22d53f292c1a189ed5ba4389bb0a5278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.29699423909187317]], [[0.27303647994995117]], [[0.294491708278656]], [[0.17155957221984863]], [[0.2873287498950958]], [[0.4887605607509613]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_86c94f99fe03736eccf5b9394d3a66fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3866505026817322]], [[0.3432890474796295]], [[0.30934077501296997]], [[0.2550120949745178]], [[0.3279913365840912]], [[0.005508876405656338]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f81e5b1632b787c44a99f4315a3603d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3970966041088104]], [[0.36351412534713745]], [[0.3172841966152191]], [[0.284310907125473]], [[0.1624995917081833]], [[0.40235456824302673]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_00eab247826505547715ad9fe9b5a46c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9df0b962f31b54be16a58bedc77ae369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_809f43676327fd8f90e01ffa0f25b1c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34900838136672974], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eea403b6d772b8f30981f23795b71f0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28569066524505615], dtype='float32').reshape([1]),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f5ed94a8d6e4ead121a49735bd6a647(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.025981973856687546], dtype='float32').reshape([1]),
            paddle.to_tensor([0.05000000074505806], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_11ea3ead4a9296c5bdea2e54cbac344d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_98f032c693720149813064303cb232e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e0a87bdd601b99adc723b6932f0ec53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6b7b5a22e95e7da56db9aa30eb6527f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_be517f13637c603cf1a4cfbdb91f1ed5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab3a64d914e8085fbb4ff7aecb391ee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4c3947d70ac17921abbc59ced9536b85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8109396594427f635a22d8bb8e1e6432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ebb93bf84f4e29068d8c3d9ea390492c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e0a87bdd601b99adc723b6932f0ec53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_173cd621f6e32e4bee7d67016e22097e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_be517f13637c603cf1a4cfbdb91f1ed5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_88e1edb798debf06d218385bb8765324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d357ac68c813293b40063e0498d3d2f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5], dtype='float32').reshape([20]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_adc491b97e2455df31eb253f563a7a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_079b021238f4ac5824a617c7f23bc5cd
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ea0e387db1ecb1cda4278eef7041573c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0119194984436035, 2.1995081901550293, 2.0441505908966064, 2.2830135822296143, 1.9193620681762695, 2.130988121032715, 2.2266786098480225, 2.1888740062713623, 1.9810388088226318, 1.9947551488876343, 2.0297651290893555, 2.211350917816162, 2.1183576583862305, 2.02701473236084, 2.1166322231292725, 2.0456244945526123, 2.1478805541992188, 2.1551640033721924, 2.150651216506958, 2.1191978454589844, 2.1087520122528076, 2.2092390060424805, 2.1003193855285645, 2.1329658031463623], dtype='float32').reshape([24]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a749992cb755bd937bd812a5c839cb35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(3.04896879196167, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da20e77694fc4c8e72ca4d9c912606a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15828561782836914], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5b7ec0037f0cbe943f59047d875f36e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_060eaee64e530bbfc175465b74c3dab0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b7ec0037f0cbe943f59047d875f36e7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0902911949774807], dtype='float64').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_43541977669dddc3df1669204fd03f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4a4c0e0f840297ae1d00580d3646da73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_43541977669dddc3df1669204fd03f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4a4c0e0f840297ae1d00580d3646da73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50bf9889e4aac2a0c803cd663dd0d1f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7780fc37244c219e8d1729848dcf68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50bf9889e4aac2a0c803cd663dd0d1f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7780fc37244c219e8d1729848dcf68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d4a5fbeb2c44550536f18aa171ea948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_608d05873d335b7ce8e7fa70a75b1590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d4a5fbeb2c44550536f18aa171ea948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_608d05873d335b7ce8e7fa70a75b1590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c7d1581b83043a11b0ab26db69451db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a40087a3eeba3718bc453b3d24697a6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_702e52fb9a9be257fb1cb0ce40b22ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_84bf68f74480f4c292f9debb832704b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a0d9baa643f27dda0d067e44439af7a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_174226ab28dfd6db584959b052e9f3b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f8b3a29de237689db1f2c859eb8fcc80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f8b3a29de237689db1f2c859eb8fcc80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_33f8d9176c9041b6a0559faef1908272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_51cee17fd5e52970df84c67b97502d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ccefedfb98ca2641caf56b6f5186e18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1547, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_df201fa7a70e22f46843a43d1387fede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c660eff2dc8015b15884ffb0aacc8edf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1547, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c660eff2dc8015b15884ffb0aacc8edf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1547, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7d9c8d3a895e65ab49641e92887602f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.to_tensor([[9]], dtype='int64').reshape([1, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7cca58149a649bbce78fa623b79e8b08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24327757954597473], [0.24460747838020325]]], dtype='float32').reshape([1, 2, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4b07e4f32cd2e941051ce6b238a2b371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dd389461351408a1c296a79a41996379(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_440ebad4b3425eeeb433852d3eeb65df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2d0d731a228ceb0a7ebf24195ef1540c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_079b021238f4ac5824a617c7f23bc5cd
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c2c552d734a31f4ba3a1b683e189d48e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([2.076277256011963, 1.9273695945739746, 2.2000176906585693, 2.16034197807312], dtype='float32').reshape([4]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1d3aa5042efeb847bd9f5da0a6032ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(0.7704319953918457, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_60a773efb857390366a1817c7c11ced1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c6b6152137ad5722034f04869a639b72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f60a70a04e31520b0a92595243fc1921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bc7fbef586c2db1acb32f4bd1ca46435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(181.84495544433594, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_be9a478727a94f5352b6750731fd7509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(4.392472267150879, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_202510cd3e909a160b4951d0eac3be36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(129.77438354492188, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_85003a50e906126ac49f7dcb4e693e3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(3.440289258956909, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a31afbed1b057bf8caef4d58a9b39a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_506e856ac880fc153219a2d4384067b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3acb1b1feeb174a0d39408a81d72fdb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(196.78990173339844, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9422e8bbceb7ad833701000ad12a63ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(7.651863098144531, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_80445a2fde87bdb5a406dc7bfe562749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08816822618246078]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_56be0e73c0f3e76c7c5a7c6ca7dbe521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15015779435634613]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e7c0287f2a8b12b4907906f32b804b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.41282951831817627]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6ae68feb52a26a6bfe375d6cbd9225c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.4128295183181763]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cfd9cf2e0bc71de97f48811cff86b2b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.09975819289684296], [-0.0024545681662857533], [0.05102035775780678], [0.03848220407962799], [-0.038630466908216476], [-0.04866786673665047]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_135718a692690fbdc3476aaaffe64586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.025793571025133133], [-0.004995339084416628], [0.02461540699005127], [0.05897201597690582], [-0.0005870102322660387], [0.013439116068184376]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e2e8ea415b56e724a2df1ba3a7263475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[-4.867560386657715], [-0.5086283087730408], [1.0727001428604126], [-0.34744974970817566], [64.8088607788086], [-4.621359348297119]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_385fb8234cb6763f9e74f586498a5986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[5.867560386657715], [1.5086283683776855], [-0.0727001428604126], [1.347449779510498], [-63.808860778808594], [5.621359348297119]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_580dd9c9a4a1b18f043349f40b352a55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e699dbcea2e3a941b73d9135c26a1f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.7899442315101624]]], [[[0.17656917870044708]]], [[[0.07220491766929626]]], [[[0.9888758659362793]]], [[[0.9760687947273254]]], [[[0.4718039929866791]]], [[[0.6848984956741333]]], [[[0.03881433233618736]]], [[[0.09308643639087677]]], [[[0.6608724594116211]]], [[[0.2744542360305786]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_def98c821b92870790446a52e96cf1a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0256400108337402], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9302c118595558cfc10bc506cf850cb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype='float32').reshape([14]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_94e4c0bc0f721de09fdbf423b1473791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5], dtype='float32').reshape([14]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a0f4f967123d907137366e0b4bf855ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -80, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c01e1516dbef05eb463760bdce770dad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0f4f967123d907137366e0b4bf855ba
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c01e1516dbef05eb463760bdce770dad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0f4f967123d907137366e0b4bf855ba
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3fa3b36bb5e62b6cd0445bd195358538(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 80, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dac62a814e9933004a6c69a607355a40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fa3b36bb5e62b6cd0445bd195358538
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dac62a814e9933004a6c69a607355a40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fa3b36bb5e62b6cd0445bd195358538
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e02da2baafd9d115b9d2c6c8c0dc855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0], dtype='float32').reshape([28]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b79237f99ddea4f23948560fa0ecbdb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5], dtype='float32').reshape([28]),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_3876c0453f51447ca5e6fc43977fd8f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -40, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0aac9d0ae13f08f0881ede3ca06c0f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3876c0453f51447ca5e6fc43977fd8f4
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0aac9d0ae13f08f0881ede3ca06c0f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3876c0453f51447ca5e6fc43977fd8f4
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_33ecc32083c24d6f8a3157023550e645(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 40, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b7fae73c0d9eb3cb269df3507fdbdf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33ecc32083c24d6f8a3157023550e645
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3b7fae73c0d9eb3cb269df3507fdbdf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33ecc32083c24d6f8a3157023550e645
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_42d2a216ca27aff230d41a4da7d53f65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bf030220603c1715613191eb855232f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_85e145a982a34a7b9e627665de8450d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -20, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fd7a2003f71bbbb61ba0f9cec254fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85e145a982a34a7b9e627665de8450d0
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0fd7a2003f71bbbb61ba0f9cec254fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85e145a982a34a7b9e627665de8450d0
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8618280eb7f319e14c1f08caca4dc14f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 20, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a8c598e76a0b90ee66bfec5c66b2465(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8618280eb7f319e14c1f08caca4dc14f
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0a8c598e76a0b90ee66bfec5c66b2465(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8618280eb7f319e14c1f08caca4dc14f
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f258574d3c66aebda2669235f6195710(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3063c79c144ad8d37d13f4e543383021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f258574d3c66aebda2669235f6195710
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_98b0c424e2b680f6f8dd446e9fb96be3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f258574d3c66aebda2669235f6195710
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_38b5fbb52acb8527563013940c5f1fa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f258574d3c66aebda2669235f6195710
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_38b5fbb52acb8527563013940c5f1fa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f258574d3c66aebda2669235f6195710
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_717f913569ef0d008e6b7c470cdd114b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.to_tensor([[3]], dtype='int64').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_25297f0c93103d251e8149158284800f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24739119410514832]]], dtype='float32').reshape([1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_83440fadc4c88a8b6bce24ac7554580a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_308ef0e251801ac6991b895c230831f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.0015625000232830644], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1fa52100f5779e0ec76d25a44ef6259e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(7.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.0015625000232830644], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ed214e410a8dec85be7285ffd3d6512f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_911ead1e22fc0b7766e496b5ed6d2e39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(186.29368591308594, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_522f9e73e0a6dd718e23ce04267522b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(3.272859811782837, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4c2af6f9fae9ad9af58d6f31ced9ab1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_361095a3f71c2c32b6fe49e88af7a358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c2af6f9fae9ad9af58d6f31ced9ab1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4f43552da2d2808abc5c274d39cad031(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_04b816ec5c9bd9a7b1398d1a355a3509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_57a21da049ed888435797cf34f1f3f94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d743bb95cc6e25c409bd4a0c52ea3873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_593464b9b0a2c1b0d759076eb9ce067e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0a44b9d58775064752d1b9f9f5a9e2fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e0eb355892419ac25f8cca3f5a9e3a29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(114.22635650634766, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c554e68e8749bb214f8abdb3a58cee67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(53.74919891357422, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_257d17f955ec3e7a9bc2e2f5fa06f73c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_11c55fc3aa1fbedaa6f7f3e7afb1210d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_866c3843d667076de7d04aa2cf9663de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_be52dd83a87ddfcc6c179240c92f11ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_be52dd83a87ddfcc6c179240c92f11ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3c1f68d3e3263e1b438a845d13b63cc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8ca3d8043604615b963e508f245ebec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e13fc49040894b4bf93bcfaff477c9aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2056, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9ad983004711dcb6b15251b1f9e837b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0ffe5a13bc08a17e817eac4fbb379d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2056, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0ffe5a13bc08a17e817eac4fbb379d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2056, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_082d5d09781440d2c0e30405088dde7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c2af6f9fae9ad9af58d6f31ced9ab1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6cb1ba982f17c3b06c3e2b222b52ef39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(189.78457641601562, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6c451552e3c86753cbaa002faf51f9fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(5.1834588050842285, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_482f8d532c88c16a95276c62fe73783f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_782b9fc5b8a90bfecfc69c3daf83de2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7faaba264af1c9778948d611e82e3b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1993337869644165], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e26911b01dad719c9edf022db0b7292c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.234736829996109], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_420a1aa40f56869f6d64bcc753547ea2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09994659572839737], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ede453659e9d1a21899055f36b86f2b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.41025346517562866], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_80ca9e381260973e83d95f2e2559f31a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4568828344345093], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b39cb93c757010ff77c3c2876aa34a04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5527725219726562], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_88ead5c4c28c6ba896d5b8e5fa2fdb12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31034454703330994], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_346d1e77f6749d9695a9558765f22dc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5957698822021484], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bab401eefa3c7c5edd03e277d1918767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06750066578388214], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_110e47e2dd6438a1fc5ed17d3e7b6b67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1016862764954567], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0baff286edb1abb2d5b701b730d80163(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.16773703694343567], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58ddacd5f74e7e7afcdf4528aa324c83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3752606213092804], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_12daa532ec90820b6d5d66be48f6223c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15381911396980286], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f38c0fbefb0af13ca02359ece649abfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5321731567382812], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a81e85e1b46e50c7415ac6e1c6b46124(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3341449797153473], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_911a5c8b4a158d3a373fa39a78d78694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3905099928379059], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9e62019c7b70a520c1dd3922b38f5bc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18431654572486877], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f214ead50419a341e2175c8c748f9f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5030423402786255], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_55438938413acfb8724b22e528cd5dfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04321257397532463], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_830433795ebfdba2a015b9961e7a8952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3429238200187683], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4acd0f04eeaed530d2780890d95079d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15142355859279633], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a6d32399a06097859cfc57a11c1d820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5988814234733582], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_efd3ff1bc54db8e91e5394d099d5e0f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ba998fa75e3760ded79ca52f868cde7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b73f3607c31326552100947327c8c199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b1ac2735327fa1fc273f07c48433202f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a80afdf8fe94397745bcbe3cdb092ed9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d09879b9b90d5731d9d276de8ee77e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f4699da7c0237c8449b4859ecbd10a
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c841ad78cb4ed7041a37884f31aeee8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_392cc03cdd87371ed3639e067a9d6f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc5dc1a9beaedf06881268b87297646
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c841ad78cb4ed7041a37884f31aeee8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b1e01202a6e0083392069ea6f2a97e2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b1e01202a6e0083392069ea6f2a97e2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c0d039f0b25e636cfd1975f4ba6ae1f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0a0b3fbb3fb4af49669ca75109e49060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0edcba4c45a189e19e2e5337994a5614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4650, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7c41a5696a1f83c21c1f34e34dd81503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8ee78f4c37d7734e0f081e5a40d5058e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4650, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8ee78f4c37d7734e0f081e5a40d5058e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4650, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_55a3bda13b7b5451c17eef816c49b983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d3c7c3e04709fec8de10a23dc32730a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(101.6871109008789, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_05d243552377189de02867137a64a05f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(308.9356384277344, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_890b35cf6f7ee74ca3ed59bea9a30c24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_890b35cf6f7ee74ca3ed59bea9a30c24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bebf67cd305440038bf6f7299f0bb744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9be989071f211f297dc6c24456ac575d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7eca85e53172ffbe208c41845ddf8813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1059, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_42b59e1ae68a8f646bed63231431aeab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cfe4ee16c5a327111211526202e12cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1059, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cfe4ee16c5a327111211526202e12cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1059, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c13708328dacab41f0a01355c8bbb4fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_82a1eaa42ef5a54afb8697c8ab3eb9a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-50.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_495a339335b160cd2f7b82456e0373b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc5dc1a9beaedf06881268b87297646
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fb56e2e92b16b4ff146707f58cb99782(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(139.52659606933594, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2e5e5a80c239fd537fb0ef0769777112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(3.771040439605713, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_517b42a79732b803632c281bf530b385(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_39075450796c08c1c910d4508cd4353e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6cfe66c26bdac34f0760014ac45b1e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14f59f8ce5eb3db3819398ef4b636788(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3b8b378d63f6b85be6be7786f95d68b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.75], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7d98b54dadc309a1ff5c26c726588ab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dc09fca21912eeca52f64e68a76e6d4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 1e-08, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4191fb812f19c588e1dc675221aadccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc09fca21912eeca52f64e68a76e6d4d
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_42eb2bcb533ebe2c175e967f6597deb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4d1ab7e4b48869d816ae55180fd8a3e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4191fb812f19c588e1dc675221aadccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc09fca21912eeca52f64e68a76e6d4d
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_42eb2bcb533ebe2c175e967f6597deb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5292001fa4205f8fe851b87cfb66ab22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_38e06df2c29b496b3dbf7ad8b0d7f4f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0aadbcea138c290bba16e4ceb329da77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aaac5b293c7bf40baf8f62d26e321117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f37c28a83410a219ff294092e973191c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_42c3133a18c12708daf0824402f30766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0f47ac513343be6093a3c857d020113b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_249caa7231d36e143853d4e41687ee54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6db970a57fd949b9611890bcd545021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7c101ec891ba57007d2f6c0136b4b328(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e431f2cf9ca1f2cd75cbabe80498bda1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f37c28a83410a219ff294092e973191c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_701b3612b7ddc0f959fc2eee0fd4c6c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0f47ac513343be6093a3c857d020113b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_17d0d4f6c3d33850d2e7bdcddfc2e96d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_028b4506162385310ffa885ade2b1fc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5], dtype='float32').reshape([17]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d3dc7b27b52743c4063fe9848612ef20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d3dc7b27b52743c4063fe9848612ef20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d3dc7b27b52743c4063fe9848612ef20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_88efda7ad8d74418d62fb0a4f5e2f20d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a9a1975c8fae2a1fdc7832d70cdc23a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.75], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a802622e19ec493d4d54ea7a92e4f6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9fbe7e69a42a4b06a0838bf41dfc782c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc09fca21912eeca52f64e68a76e6d4d
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_820b24f69d45a71d4cfeae3a2638d61a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7f922ac68fcd5f1e137e7c999af38914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9fbe7e69a42a4b06a0838bf41dfc782c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc09fca21912eeca52f64e68a76e6d4d
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_820b24f69d45a71d4cfeae3a2638d61a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ed40f7fcfbbbcf1da43945b8ca089d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.12368782609701157], [-0.034869760274887085], [0.012797508388757706], [-0.07258497923612595], [0.003080888418480754]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1864f91a00d70f72f45065dd4426e311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0032057890202850103], [-0.02368786185979843], [0.014531121589243412], [0.02121308632194996], [0.02454344928264618]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b278e4d8244bd085798abf9b41db4d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[-39.582645416259766], [0.47205182909965515], [-0.11930346488952637], [-4.421707630157471], [-0.8744720816612244]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9044c7d519714bf05352cf87ca26ce08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[40.582645416259766], [0.5279481410980225], [1.1193034648895264], [5.421707630157471], [1.8744721412658691]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4260793f5e0cd1c471dd98eb694c4e34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d4a5fbeb2c44550536f18aa171ea948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_608d05873d335b7ce8e7fa70a75b1590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d4a5fbeb2c44550536f18aa171ea948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_608d05873d335b7ce8e7fa70a75b1590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50bf9889e4aac2a0c803cd663dd0d1f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7780fc37244c219e8d1729848dcf68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50bf9889e4aac2a0c803cd663dd0d1f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7780fc37244c219e8d1729848dcf68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_43541977669dddc3df1669204fd03f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4a4c0e0f840297ae1d00580d3646da73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_43541977669dddc3df1669204fd03f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4a4c0e0f840297ae1d00580d3646da73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3a7bff229d2d7775525f81c70d888037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_909c93802435aa86db301e86428d9b68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3a7bff229d2d7775525f81c70d888037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_909c93802435aa86db301e86428d9b68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53e0908d0adbaa226b395ee6e5e85f55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2075a466fe5b042bf841ab6407e5227e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53e0908d0adbaa226b395ee6e5e85f55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2075a466fe5b042bf841ab6407e5227e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eec7d00ff6a2ac873cd74105616ef56f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0df0508c92033ee0f75d944e608c785b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7ccd26fa2fe375cdd29a35d8a497e507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7ccd26fa2fe375cdd29a35d8a497e507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7ccd26fa2fe375cdd29a35d8a497e507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3a244f62f4b7fd18d33aa83cce7391dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dbd9567b5ffcd7bb4deabf8f484742f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f49944d60d8441d92036f280bad3eda2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc5dc1a9beaedf06881268b87297646
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dbd9567b5ffcd7bb4deabf8f484742f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cd61a397e033fb21143823daebfc46df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a9319ac17177c0eaea9d2144680f5c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aaaf7730cffa8a3dac82578618ca2ad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0842bece60a0229c7a84d8b7191886db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_96951fcc8f43cf8939c89204dbbdb574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3aac64d1d870e4aa485bfe201219796f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab370fb04b8afd7a94200140669ded60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19193267822265625], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cb129d0613ca1e0255eaeaab681c2b8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4994480609893799], dtype='float32').reshape([1]),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da3a0342df561a635719014a9619c00d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23286379873752594], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b7f44166b2441af56bc80c3b0ab5fcae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bc8b568168b40384c20003e735f11789(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfe93d91c810a0df87ba53736e1e0d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8d040b154d772af8ff058575273b578b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8d040b154d772af8ff058575273b578b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c491fcb7968273d4b8052ac34eedcd2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7c7e8555d94884abf8ab909222937f8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa6a5b0fe198e789570105247ea4f4a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2347, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_602cf22c5d9321099df1fbdf116a4e54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_afa0f917140f624747f8fefc33a5fb26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2347, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_afa0f917140f624747f8fefc33a5fb26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2347, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2d4094da0f8d71a8d0bb8f777f39d3fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2d4094da0f8d71a8d0bb8f777f39d3fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3ba80b02772f2ad0bb114678f9228dc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_05e6f0e2c3301c11f4d44da530ddfbc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dcc1779057dbc77b52105857b16c7685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3109, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bd84a1aefb2da6aa55b94327af39e472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4c562f3a7c68a978fa43abe3b3538754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3109, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4c562f3a7c68a978fa43abe3b3538754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3109, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6eb1aec4f2900aec847f64a80ca7ac77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6eb1aec4f2900aec847f64a80ca7ac77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6c72df5bf0e04f08458d13f17a6bc997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ca954c64303df615c512176112efd87c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_15d5dd43f3e30d43dbb42708896252ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3813, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_98f7bfee53860e887b75b465f0a5c50b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_426d578e5d32bb4e763cb582f16aa607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3813, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_426d578e5d32bb4e763cb582f16aa607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3813, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b53885157c45e52ef1b4467a848c5499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_180d0cdfe11f1046f5c1e891007fac6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc5dc1a9beaedf06881268b87297646
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b53885157c45e52ef1b4467a848c5499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_81d3c7c1e246752099b888b3babfa3da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fbb5c954da434dae27266b071e36d3ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4b22ee5398e2e1774f5d0f5241b11f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f4699da7c0237c8449b4859ecbd10a
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_044ca33194a1ad4c4c747ca504e49bdf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 0.925, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d4268155c83130709b87ffa439bd035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_044ca33194a1ad4c4c747ca504e49bdf
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.3623609244823456]]], [[[0.6762513518333435]]], [[[0.7048158645629883]]], [[[0.6515768766403198]]], [[[0.43987032771110535]]], [[[0.6021124720573425]]], [[[0.7940725684165955]]], [[[0.14514058828353882]]], [[[0.11087203025817871]]], [[[0.8659011125564575]]], [[[0.12744580209255219]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5adbd45af46fb625e87314777e05f4b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0810799598693848], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_efd3ff1bc54db8e91e5394d099d5e0f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6e36dd3128da4383c26ed04928f1219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7a802f157fcfd3d58f44506227283157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6665d9271575d5e405b06bf2bd43913e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_54c0e9dc890f2f99c562c10899b161ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f8d02d482dd29d10689681afb2c6f2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ddfa1f676268a5ca8351132182d70f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_29781797919706594de6780cf4a88fda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b75f6faafc9237011d6efa861137c446(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8aba2c2b79f05f6aa92c3e6461da8f0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_961c3e77559336f0dcbaa37e5a713bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aba2c2b79f05f6aa92c3e6461da8f0f
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_961c3e77559336f0dcbaa37e5a713bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aba2c2b79f05f6aa92c3e6461da8f0f
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c2b6f333b37f7930d18d4e6b58bcab6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e5039559b9cf5396a3da8b0f1e56401c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2b6f333b37f7930d18d4e6b58bcab6c
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e5039559b9cf5396a3da8b0f1e56401c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2b6f333b37f7930d18d4e6b58bcab6c
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e13fcc0583e64a72d8fed8aed9542dc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9f43b7505964542d30db0ce4158f9b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e79d1a581d16939cab72d5017603fc29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f35b420807b2c382e3b5944b901ae80a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_008c028cb7e50589906d568dd2a77349(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -64, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7b912bf635952abe08cd19c09911c6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_008c028cb7e50589906d568dd2a77349
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e7b912bf635952abe08cd19c09911c6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_008c028cb7e50589906d568dd2a77349
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7488700375ae67f959415fc79a3f112c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 64, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e00ac422f4e246db0d525472d1e2da7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7488700375ae67f959415fc79a3f112c
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1e00ac422f4e246db0d525472d1e2da7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7488700375ae67f959415fc79a3f112c
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d08ac33677428c3e9615f41bfd9f6690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7131620c22be27824465d6ab33e21549(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4ea72f998b2ea318897c3348377f520d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0], dtype='float32').reshape([25]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7a8eb45749c72e799ce6a77a822923b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5], dtype='float32').reshape([25]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_37776f80c503717897db4af0a8c16006(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -128, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27285836175c6fccea9cf90f308ecb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37776f80c503717897db4af0a8c16006
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_27285836175c6fccea9cf90f308ecb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37776f80c503717897db4af0a8c16006
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7cc86a8c6809b87f325d839d78737131(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 128, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7551dcfb855393d987e1578eb98fc513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cc86a8c6809b87f325d839d78737131
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7551dcfb855393d987e1578eb98fc513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cc86a8c6809b87f325d839d78737131
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f5ab9920ca4a3d14fe820980ec521e30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], dtype='float32').reshape([19]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a38bbbfbddb694a63a873212afa1941d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5], dtype='float32').reshape([19]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab60060329b888643900e389521b055a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype='float32').reshape([13]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10597b470f41d473a75425a1f5e2b6b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5], dtype='float32').reshape([13]),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9d97f9438fbd1839f4e4f1c8d8047477(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -256, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32ee3cbfc0873d77920bd1046da48429(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d97f9438fbd1839f4e4f1c8d8047477
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_32ee3cbfc0873d77920bd1046da48429(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d97f9438fbd1839f4e4f1c8d8047477
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_67271733891d5bd01ade7457bafc00cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 256, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a92764cc9eed0cf89a297416f5445a98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67271733891d5bd01ade7457bafc00cc
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a92764cc9eed0cf89a297416f5445a98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67271733891d5bd01ade7457bafc00cc
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_18a76a73a3df5c757f41aceb8f943d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype='float32').reshape([10]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_91a21edb6a993be7df96ab9633feb7a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], dtype='float32').reshape([10]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_be48f8096ce31ed3b9d52335038fe1cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float32').reshape([7]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_28fe1eb060da69cedef47fcc6a040485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype='float32').reshape([7]),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_483c53c097ba6b917cbd2d8002a00fb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, -512, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e9fb9e654aaf618f3f0c9e1a1eef986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c53c097ba6b917cbd2d8002a00fb7
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3e9fb9e654aaf618f3f0c9e1a1eef986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c53c097ba6b917cbd2d8002a00fb7
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_91dac0b3ef651da0145a53c05f97b419(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.scale(input_0, input_1, 512, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e3515ea5194c71f8dd7c4c6660e8ef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91dac0b3ef651da0145a53c05f97b419
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e3515ea5194c71f8dd7c4c6660e8ef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91dac0b3ef651da0145a53c05f97b419
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4f313088b93a400e107f523bb7461105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0fc1ad90ad83c6533a8daa033c42ddb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9ce8f5a1a602cd4c2bc0b37c996173a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_079b021238f4ac5824a617c7f23bc5cd
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5ca8d0ae8264491e89a2bb1b0887472d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0498390197753906, 1.9718046188354492, 1.9871633052825928, 2.2192506790161133, 1.8808510303497314, 2.111492156982422, 2.0883917808532715, 2.2100749015808105, 2.206063985824585, 2.153613805770874, 2.0428218841552734, 1.954197645187378, 2.005186080932617, 2.1854147911071777, 2.1777029037475586, 2.3102617263793945, 1.9677259922027588, 2.123687744140625, 2.02335786819458, 2.1780855655670166], dtype='float32').reshape([20]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_37897e2e52601d06221dc738b11cbbd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(2.738473415374756, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b389bccc94f806e9cce23dd8ab313afa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_37d67445de2525a8cdcf0a9e47daf858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_43541977669dddc3df1669204fd03f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4a4c0e0f840297ae1d00580d3646da73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_43541977669dddc3df1669204fd03f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4a4c0e0f840297ae1d00580d3646da73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50bf9889e4aac2a0c803cd663dd0d1f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7780fc37244c219e8d1729848dcf68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50bf9889e4aac2a0c803cd663dd0d1f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7780fc37244c219e8d1729848dcf68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d4a5fbeb2c44550536f18aa171ea948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_608d05873d335b7ce8e7fa70a75b1590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d4a5fbeb2c44550536f18aa171ea948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_608d05873d335b7ce8e7fa70a75b1590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3f06c7e46435e28da4b77fb067ba3ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(356.6340026855469, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_18079c90150202a5cf1f2a19a6d4cfcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_49416be2f7bb5a41463b252b1d666c29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.010650492273271084], [-0.027310103178024292], [0.0060086315497756], [-0.07410187274217606]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ecff08e35aa9ef6f8c77a8e157cff7fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.007471632678061724], [0.006812167353928089], [0.021765010431408882], [-0.030457444489002228]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_750e1904d748f70613e37d8bcc2c31ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4254571497440338], [-5.0090179443359375], [-0.7239316701889038], [1.4329642057418823]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_631150de25caffd2f6ad49cbc8e2b5fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.5745428800582886], [6.0090179443359375], [1.7239316701889038], [-0.4329642057418823]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b7ac3276ced650c9d696d9d502e830b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_591f6bcd8a492097f9836dfd6f30c3e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(33.97206497192383, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_537f377a878bb183cd231e01da8c136d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bd00524ac898769d2f7398c510382e4
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_850da97f3503448cd038c9477ffa1912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0526299476623535], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09e7ae68d59afdf4414c530586c1ea5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09e7ae68d59afdf4414c530586c1ea5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c65801c8daaff15d4e6058a9e6aa3289(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_81af85b7231ebb7cbd4987cd99e973e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8221ca1054d56e1d6847a1ce3665ee82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2100, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ff6fec871d23e609e954a360bda4160d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_654dbe7daa6a1833d10448587227af10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2100, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_654dbe7daa6a1833d10448587227af10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2100, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3e743a7bcaf2dd4598bb82964e73a948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f1b0cd832c0f48e4d174c81a0de864c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b6a08a398554880e468f1de7acc90983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_19abc28f5231f94ca5bdcda07b2ea989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ce1ac859b0b7de825074abfe9496659(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1925831f920aabb39ee59d98d5a5e418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_799f0bf7359fdd55ff291663bd30b861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc5dc1a9beaedf06881268b87297646
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1925831f920aabb39ee59d98d5a5e418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7ed73923122a1b382e78019454cde307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_61cf966e874e9297927308bf4506e0e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5cd530d464c093499d31251b8d7471f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_79023d8e354af0c1fbd55465231c3cb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ea3d6fe1be2990d480b6cf9c3303916f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4da0cbc0016be3e87b4d38066894eb7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_21d9f9431ae0ddd5dfafa7d7a95a511a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2151b93c5afb9fe93d71e5dc83d9cfec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d9d0b233dd9c64e12c7589ee5b6d276e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_64fbaf3e9d9b5de0ef5a104983d70108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_79023d8e354af0c1fbd55465231c3cb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d5115ba435fb85197abddbbed4a9dad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4da0cbc0016be3e87b4d38066894eb7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1631cef915d9928c2dcd561653b60e9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9700d22b586cbc59e4755d9ad63a1e95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5], dtype='float32').reshape([18]),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e3e803f32b8ce7b3b91d318e3022ac17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(132.92568969726562, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_133fc8176b11504339a48975212f23d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(2.5554423332214355, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_96951fcc8f43cf8939c89204dbbdb574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f709f19f6b8dd33d8341f6ae17cc2e45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ee1015d5352237d0df5e12940ced4483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_015e3647e32268a6188174bb84e29ea3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9092d694a15b38ebc4b199b9d4c5dc44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ce38fad95e06bc1f0fed7673c5913962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7ccd26fa2fe375cdd29a35d8a497e507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4c591cb5d6f362f95e59a770be9148ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_11ea3ead4a9296c5bdea2e54cbac344d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5d66d8bb827c590080899c3c8ed3b024(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a9ff016d1a1cfa1ae8b1ac3b88c60419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cdf0980705ddb261e1870773623681c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b0e4302b937ffdf6c14fb08505648b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(36.244232177734375, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_91dfb11b32a0dcf2539c3417aa4269e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_422292d2e13edfa25dfc83aa2b9b5f07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d6a8aba282499182ddbeb2668db3e863(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d6a8aba282499182ddbeb2668db3e863(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1eecc7463998c8c082bcb3bc82d14ba9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9c0071bc0e48c3c7af73bfb6963de714(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58fd71c86d550e25a99bc82f3322221e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4231, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f27525a07ff1d6c53fa8c348804c4237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_93eacb95eac2cfb286c3576cdeb6401a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4231, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_93eacb95eac2cfb286c3576cdeb6401a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4231, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_483f8232c0b49b1b2939d1985c176da2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f539564d407b2307f6e6df6b1e379a41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_543e436ec9dca0138f3187f58c74b405(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(98.78099060058594, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_15f8b5abcd0335b3a845c4bbe0c48d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(3.242335796356201, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_43541977669dddc3df1669204fd03f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4a4c0e0f840297ae1d00580d3646da73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_43541977669dddc3df1669204fd03f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4a4c0e0f840297ae1d00580d3646da73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50bf9889e4aac2a0c803cd663dd0d1f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7780fc37244c219e8d1729848dcf68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50bf9889e4aac2a0c803cd663dd0d1f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7780fc37244c219e8d1729848dcf68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d4a5fbeb2c44550536f18aa171ea948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_608d05873d335b7ce8e7fa70a75b1590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d4a5fbeb2c44550536f18aa171ea948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_608d05873d335b7ce8e7fa70a75b1590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_42d98b2eb837536d570d9ca01ee80f09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_91dfb11b32a0dcf2539c3417aa4269e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b62d69b67d78a26f1667d9c4d30751d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f18e14835d87eb29b60993f7373e5068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2c522f774c0c88b47799fd7cde607ce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(34.44670486450195, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()