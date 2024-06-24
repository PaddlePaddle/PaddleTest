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


class TestPrimitiveOp_4400fc318dfe9d6b278a6abef2be3cc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92cb719a58e611bb95c2eeb1f148340f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.08261148631572723]]], [[[0.4215710163116455]]], [[[0.7736701965332031]]], [[[0.9594077467918396]]], [[[0.21585646271705627]]], [[[0.13334153592586517]]], [[[0.8511208295822144]]], [[[0.8058916330337524]]], [[[0.7008373141288757]]], [[[0.8254395127296448]]], [[[0.6122303009033203]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_b52ec14f5e3e8e05f594cc4017f491c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.3148560523986816]], [[1.3181918859481812]], [[1.1191445589065552]], [[1.0070120096206665]], [[1.1452118158340454]], [[1.0538450479507446]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dca5d03cda46de061ddc8e535508079c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.060086727142334]], [[1.1225395202636719]], [[1.4771132469177246]], [[1.4491592645645142]], [[1.442306399345398]], [[1.1255842447280884]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_c0f71f289cbb27e930c1a5c38323ba1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.46424782276153564]], [[0.16565507650375366]], [[0.24447554349899292]], [[0.4760080575942993]], [[0.25089192390441895]], [[0.32649412751197815]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a122801dec8baa10e7687f2442e5e2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.4705996513366699]], [[0.09719990193843842]], [[0.17953190207481384]], [[0.08616776019334793]], [[0.4652281701564789]], [[0.18960219621658325]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_501fdb044e92439b11a03263cb85c41a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.42634397745132446]], [[0.16619716584682465]], [[0.07921028137207031]], [[0.3810620605945587]], [[0.40618160367012024]], [[0.2721419632434845]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e4df23a547315ad04a54c25c72996d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.35580918192863464]], [[0.37486857175827026]], [[0.42684468626976013]], [[0.0039510601200163364]], [[0.3080861270427704]], [[0.15155068039894104]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_651b6f72267621502a2a1fbd87f245f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24425794184207916]]], dtype='float32').reshape([1, 1, 1]),
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


class TestPrimitiveOp_7992bbb5db82013771253a4b3b988151(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(1073.7626953125, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f5977ea244baed7bfa801bcbe20bcec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(181.55203247070312, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b9fa7841cce5d344e5d37bdd92adad62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(6.338008880615234, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_17d03cadc21dd5791d4a541012b3ca57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0010424084030091763], [3.965394716942683e-05], [0.0068883392959833145], [0.0026569680776447058], [0.007810906507074833], [0.0021921945735812187]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d0b663bbabbe6d8d377f96816773a360(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.008773252367973328], [0.01485530100762844], [0.009202161803841591], [0.0006451195222325623], [0.004761107265949249], [0.0028145231772214174]]], dtype='float32').reshape([1, 6, 1]),
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


class TestPrimitiveOp_451d775a37497dcec869419b0849eda5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2840380370616913], [0.33663278818130493], [0.07468066364526749], [0.24954964220523834], [0.22653621435165405], [0.1327381581068039]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([0.08333329856395721], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58894829d35fe9bf7f33dbbf358efe92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.023669825866818428], [0.028052721172571182], [0.006223386153578758], [0.02079579420387745], [0.018878009170293808], [0.01106150820851326]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([6.28318977355957], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_22b68d16f3552cc134b44fbfe68402f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.14872200787067413], [0.1762605756521225], [0.039102714508771896], [0.1306639164686203], [0.11861411482095718], [0.06950155645608902]]], dtype='float32').reshape([1, 6, 1]),
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


class TestPrimitiveOp_abae515eca83da781fb85802dbfbfab9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([2.073956251144409, 2.060838222503662, 2.041071891784668, 1.9943456649780273, 2.016728401184082, 1.9964838027954102, 1.9643328189849854, 2.049067497253418, 2.1503772735595703, 2.075239896774292, 1.8712064027786255, 1.984096884727478, 1.890690565109253, 2.079070806503296, 1.8683030605316162, 2.1427664756774902], dtype='float32').reshape([16]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1f103342a9f0f15004584038b4b0e08e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(2.170518636703491, dtype='float32').reshape([]),
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


class TestPrimitiveOp_f3e5002a4586136efe46c52548eb5aa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(34.65229415893555, dtype='float32').reshape([]),
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


class TestPrimitiveOp_59fa6009f74816d27ebe23377709ed10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59fa6009f74816d27ebe23377709ed10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_3445edb052de98bdc83b2892bd99750d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_55ceacbd50b4055cea393be134f5054a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_bc9f74d41974b1ddddc47c8af5089ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1787, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6517accd92cf0de011475cd2e4fdcec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1787, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f5b7a6a8f3b4ede8e707ed1b5383747b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1787, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f5b7a6a8f3b4ede8e707ed1b5383747b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1787, 4], dtype='int64'),
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


class TestPrimitiveOp_56521478dce3f5986d267ddb648e8998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18c64c64a19beef4e5afcb80a70fb2b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2686673700809479]]], [[[0.35604339838027954]]], [[[0.08268909901380539]]], [[[0.8983410596847534]]], [[[0.928766131401062]]], [[[0.21594786643981934]]], [[[0.1092219427227974]]], [[[0.3836325705051422]]], [[[0.7382863759994507]]], [[[0.28424516320228577]]], [[[0.4547399580478668]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_835fb557d063e8d51e57a21094c68205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bd00524ac898769d2f7398c510382e4
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0863569974899292]]], [[[0.07638834416866302]]], [[[0.6133827567100525]]], [[[0.9024407863616943]]], [[[0.7439473271369934]]], [[[0.6636244654655457]]], [[[0.8081923723220825]]], [[[0.5607835650444031]]], [[[0.7601913809776306]]], [[[0.8342726230621338]]], [[[0.7765789031982422]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_8032346b09117f9fde900afcd1bcf75c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(110.49693298339844, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2d792bc980fccee0d1a6181d5771276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(2.9386911392211914, dtype='float32').reshape([]),
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


class TestPrimitiveOp_3446e0851959f5883861e3f22e15a5fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(159.8219451904297, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9bc2bc1dba31bedc835a919405ee27df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(3.4601964950561523, dtype='float32').reshape([]),
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


class TestPrimitiveOp_050bb5e8417f84b130f81927f1949a5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06219828128814697], [-0.014282139018177986], [-0.10170114040374756], [0.03550627827644348], [0.019529074430465698], [0.08100476115942001], [-0.07937122881412506], [0.03876090049743652], [0.0033347271382808685]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2ad98751dbf3a32e06b87713444f263c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10067952424287796], [0.035073231905698776], [-0.06951171904802322], [0.06708165258169174], [0.016296496614813805], [0.00010452861897647381], [0.022295046597719193], [0.007817914709448814], [0.004220306407660246]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6dd967fcab0146917d70f34be075efc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3374488651752472], [-1.4072091579437256], [0.4630790650844574], [-0.47070059180259705], [0.19836029410362244], [773.9521484375], [-4.5600385665893555], [3.957958936691284], [-0.2098376750946045]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c135aee24ce00fa89a3507518e54aa8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.3374488353729248], [2.4072091579437256], [0.5369209051132202], [1.4707006216049194], [0.8016396760940552], [-772.9521484375], [5.5600385665893555], [-2.957958936691284], [1.2098376750946045]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_026f6ed2742731bfd48584fc06d698c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(11607.1171875, dtype='float32').reshape([]),
            paddle.to_tensor([0.09090910106897354], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a25bdebb882531be450c57a0c943b368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(1055.192626953125, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c8bd0effc23a6e9f9134e954f4ac9f11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35690101981163025], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f87eed45fd0e24444caa92d938f5e793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.1386245489120483]], [[1.1945123672485352]], [[1.004612684249878]], [[1.1472545862197876]], [[1.2108033895492554]], [[1.455592155456543]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_88bed60b814a0073cd270f89161ece5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.4026798009872437]], [[1.3140790462493896]], [[1.339162826538086]], [[1.463433027267456]], [[1.057719111442566]], [[1.1734230518341064]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_58aea7a620a5db8e9b6e444a10f40c55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58aea7a620a5db8e9b6e444a10f40c55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_175af4a525ba60e5f65a89e3fc8370c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c20c932384a758604b8fa10a05b44032(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c9f70b1a8bb5fb5a58732b758e127316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5524, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bd7f7802a1b90879733ebd4882b3a1ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([5524, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2ef03dd50f509e26282b99a4d63e77f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5524, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2ef03dd50f509e26282b99a4d63e77f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5524, 4], dtype='int64'),
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


class TestPrimitiveOp_bbc0e592c47dfb4ac301f81f240f6400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(167.3988037109375, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bb917651ab14a9caa4473b231ceba683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(5.193711280822754, dtype='float32').reshape([]),
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


class TestPrimitiveOp_de689e5c02fa227cd4074ddaa1b58912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(7.2180376052856445, dtype='float32').reshape([]),
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


class TestPrimitiveOp_7ad379519360adb8811f97961174171d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4013482332229614], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d73f8f00eba9eab087ace422b0e8889f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22211085259914398], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9a79b999181b307a0518ebdb4d0cca4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.804855227470398], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_68936970f94c6e8991494a36021e51e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(137.02940368652344, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_449f080118bac8b5be273e515eb3a65c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(58.68014144897461, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a38890552be0d903016f44285f7d05c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14229030907154083, 0.45878663659095764, 0.5658540725708008, 0.2984488606452942, 0.3893716037273407, 0.2647029757499695], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_767dfaca5369cd41700d4a0ec5dc773f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5333330631256104, 0.28519368171691895, 0.4679723381996155, 0.672787606716156, 0.579139232635498, 0.6617571711540222], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5ecacc029238ef55ca0c0949d0a6c868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5140538215637207, 0.9951491951942444, 0.6413029432296753, 0.28409847617149353, 0.7639040946960449, 0.4888470768928528], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_42bd2ffbc4584d2a7380b97dc64bd4cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7031620144844055, 0.45973527431488037, 0.5822639465332031, 0.9029615521430969, 0.34026503562927246, 0.26559531688690186], dtype='float32').reshape([6]),
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


class TestPrimitiveOp_02cf691a2e123f4bca8048e48c6be575(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6e893b1e74b938d469fd1956bd5fe0
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.0919826328754425, 0.0004508086130954325, -0.020791757851839066, -0.0002540444256737828, 0.01552492007613182, -0.0659312754869461], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f778a2b168e1e8a192cf98be0bdbeb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6e893b1e74b938d469fd1956bd5fe0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.041762497276067734, 0.07953738421201706, 0.004688776098191738, 0.01329649519175291, 0.04933386668562889, 0.05179619789123535], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7cf588d7bb99da8c5f996be002cef40a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6e893b1e74b938d469fd1956bd5fe0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06237894296646118, 0.11125954240560532, 0.12761320173740387, 0.0912887379527092, 0.08598051220178604, 0.140384703874588], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3839fb42687ad909581e17f901322e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.2194795608520508, 0.6516017317771912, -0.8472384214401245, -0.48004817962646484, 2.033503532409668, -2.1933774948120117], dtype='float32').reshape([6]),
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


class TestPrimitiveOp_7a54a66b6a9deff97bde09af4a595b25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33fc97b08736fb3dd9b6dba9d150c721
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.0, -0.0, -0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d94a2a5b481c9adbe243377907476178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6e893b1e74b938d469fd1956bd5fe0
    def get_inputs(self):
        return [
            paddle.to_tensor([1.6027116775512695, 1.1720778942108154, 1.2909188270568848, 1.0933964252471924, 2.6759088039398193, 2.9497876167297363], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_44a6dfa09cb6cb218913e6dcf108f63a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([1.896150827407837, 1.740144968032837, 1.1023029088974, 1.1536309719085693, 2.6233930587768555, 2.6577539443969727], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b333357a3fee0109a8527cc8eedca5e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(1.8622294664382935, dtype='float32').reshape([]),
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


class TestPrimitiveOp_d6b909bac97f55839af210274ef61975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d6b909bac97f55839af210274ef61975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4b1a0131ec1d671b47014a477c95a0d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_332c1b20e8d61d9f8814d3435c2013b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f408f1b68e65a1b1aa4a7eaf477f9538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1722, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c32bd98f43f23722a5ba2bd7868b807e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e1e78f249786e27b8bd6af8cc855f53e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96a0c65442ab89e3c53601857861ab7d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1722, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e1e78f249786e27b8bd6af8cc855f53e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96a0c65442ab89e3c53601857861ab7d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1722, 4], dtype='int64'),
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


class TestPrimitiveOp_12a15550030a5182024c16bdb63d1535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.13873741030693054]], [[0.13893212378025055]], [[0.2273373305797577]], [[0.17077070474624634]], [[0.2892541289329529]], [[0.25426748394966125]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8e4970faf99c4f18409673f065f2809a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3854258060455322]], [[0.18957515060901642]], [[0.17694276571273804]], [[0.0017483800183981657]], [[0.26534363627433777]], [[0.07485723495483398]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_99a9f62456a2dc441d2f20a200590874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.18275712430477142]], [[0.38497790694236755]], [[0.13323169946670532]], [[0.19547641277313232]], [[0.37189048528671265]], [[0.42880791425704956]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b1250c21e250dbc3428996b3e62ad37d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.43969258666038513]], [[0.3787880539894104]], [[0.4903319180011749]], [[0.11881936341524124]], [[0.18562017381191254]], [[0.12537863850593567]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_7f9af1ed8fc0bec2ce6613572d0427d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3366888165473938], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_39c05faf425afe02e9392cc2bf29cdda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03960176929831505], dtype='float32').reshape([1]),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_084732346650a5961e030a0b59c576f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4740322232246399], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_11551ea6bae95485280d8c16dd125084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9130730628967285, 2.000300884246826, 2.1497082710266113, 1.9417407512664795, 2.007229804992676, 1.95833158493042, 2.042038679122925, 2.079796314239502, 1.9896266460418701, 2.075540542602539, 1.9539878368377686, 2.196959972381592, 2.176316976547241, 2.0061745643615723, 1.9684706926345825, 1.9651684761047363, 1.9509186744689941, 2.0736045837402344, 2.1274948120117188, 1.9405628442764282, 2.1291584968566895, 1.9361212253570557, 2.069420337677002, 2.172781467437744], dtype='float32').reshape([24]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_028fb32f36457451c856552f4bd1fd19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(3.075927257537842, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9f183c80925248bc58554b9272daa42d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19073332846164703], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_1891433fd81aa1b7dedccd4413740401(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b7ec0037f0cbe943f59047d875f36e7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35042174715262686], dtype='float64').reshape([1]),
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


class TestPrimitiveOp_fae5fa7da83e790bb694621b4dbc95dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fae5fa7da83e790bb694621b4dbc95dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da06afbd3d66144f49e25d60d6bec118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f2579550a78ee767e70b7b8f5508428b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5c6dfaa8ee125fd86f79da4fb8e72cbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1565, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ff82c6b9c985a9644e35dd29b039edec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1565, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e13bef0c14162e57eeaf4b99e044f74e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1565, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e13bef0c14162e57eeaf4b99e044f74e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1565, 4], dtype='int64'),
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


class TestPrimitiveOp_67a722b6c39250aca43b084cefc945f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24382051825523376], [0.2482498586177826]]], dtype='float32').reshape([1, 2, 1]),
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


class TestPrimitiveOp_1db78700fa63d8a21e782b5118485854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([1.8913545608520508, 2.063997268676758, 2.201589345932007, 2.067262887954712], dtype='float32').reshape([4]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1eb2c7dad8e7301a31e509ec48527af4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(0.7418696880340576, dtype='float32').reshape([]),
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


class TestPrimitiveOp_83af8818daff6aa2159ab30ec2055dd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(148.80516052246094, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f75fa9479682c35d9b686274303f3f41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(4.598909378051758, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_695ece0d40c1f5384aa25b9a38d98768(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(175.77870178222656, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_680685e54ab461fce9abdd9dcca032df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(3.486360788345337, dtype='float32').reshape([]),
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


class TestPrimitiveOp_55c699eff6389ee18035ebd7b8bea790(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(180.0463409423828, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0b4b8970eeedd457e5a812688cba1634(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(5.939418792724609, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c90804bf03a503ffe377ee5c6d5e8e18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.020566539838910103]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4a080bd25f4f68b9efce21315e2fc96a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12849251925945282]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c9a071bce516cdd8f6db967eeea184e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.160060167312622]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4c64ac5e27e7b49cff7cddcf84b64645(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.160060167312622]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_98c62f57ba80f706be2c69d02613a539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.01087585836648941], [0.013842154294252396], [-0.04264052212238312], [0.06212622672319412], [-0.0019288919866085052], [0.022789228707551956]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1d91104dd9e3b2a6fa9fb9e164e06161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.03250936418771744], [0.058684736490249634], [-0.025082498788833618], [0.009060210548341274], [0.023795899003744125], [-0.015243959613144398]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8e89deda71deaf85a0063c7568d9270b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.334545373916626], [-0.7641268372535706], [0.7000109553337097], [5.857039928436279], [-1.0810598134994507], [-2.4949676990509033]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ceebad2f11a43fcd39b7beb2700b847d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.334545373916626], [1.7641267776489258], [0.2999890446662903], [-4.857039928436279], [2.0810599327087402], [3.4949676990509033]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_249f4585611a31be3c2bff92df917ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e699dbcea2e3a941b73d9135c26a1f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.43089643120765686]]], [[[0.9136064648628235]]], [[[0.09930074214935303]]], [[[0.7008950114250183]]], [[[0.9837452173233032]]], [[[0.7385526299476624]]], [[[0.24156886339187622]]], [[[0.679689347743988]]], [[[0.3776068389415741]]], [[[0.4095267653465271]]], [[[0.6420373320579529]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_b7cc30096e09df739ab3e5b6fc6e8261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24200740456581116]]], dtype='float32').reshape([1, 1, 1]),
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


class TestPrimitiveOp_e43e45dde05deb59d08c6f4a46770513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(169.502685546875, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8a62f7ed4dc472cc49e1d203c7df7eb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(3.36458420753479, dtype='float32').reshape([]),
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


class TestPrimitiveOp_8505ec80ee986349f176dccb02f6a59b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(121.32744598388672, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3e55072b4081a660fc32f694096ab542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(43.08905029296875, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6db0c4c0535489cfa0964e7ef87b30d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6db0c4c0535489cfa0964e7ef87b30d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_87b2aa5f5d70c7a0894b188806ab4d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4fde8e371490b0691f16f22f4e491c43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9e22351155e6366a9e38626fa8c77baa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2034, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_52ba3fecde000efe4ad5f86e1473a5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2034, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10c9d27c3850db5e04bd7bd24f5d0d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2034, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10c9d27c3850db5e04bd7bd24f5d0d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2034, 4], dtype='int64'),
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


class TestPrimitiveOp_86ac43bd36633260d2a815f88b2046ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(184.30282592773438, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76c93dd70beed92156f0eb4a2c7b666d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(4.240999221801758, dtype='float32').reshape([]),
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


class TestPrimitiveOp_d284a00ba81ea2a83c3032f86a6a5ed5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13775603473186493], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_17a41161519b84bcfd694c935ef021ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2208438515663147], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2be0849652d8a240e5e9bb38a5f33c36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22658437490463257], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_be7dc9d59eaf4fecb4d85fb2e3a9bd5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2321021556854248], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_070baed3cfd2ae1b23ae12aeb6138127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.051429517567157745], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10cd377a98a0f02316bc15f84cd1d0cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4713531732559204], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_238d78cba1324f74c075ee7425c4f5fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23564143478870392], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_65b0dd87e3d95d267c167fa22c86cdc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27569863200187683], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a87db5d7d38c5070f3c2beb9c3ca4136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1636916697025299], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6c3569174b8824b0787e8314f5697800(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.36039450764656067], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c1d87b4a8a033b701499ca06c12f3702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09617427736520767], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e36a687fee5f263c1f9dfbc51c3b4975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1586979329586029], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_57800a740d206dfdf474b2cfc91cd703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21375669538974762], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_942204cb0c5053060abc5cda46831fc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6157842874526978], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_535b70648f800056998e21afc4c08bec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.025367464870214462], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_25b9148d4e8bf6c4ac4e41ce24e0e3e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4182332158088684], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_666a9cd3ab2037a053671b7b742ed6c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05307089909911156], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8b7c5ca287dad1cfc98cfd11f74ea185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12740199267864227], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_286d8ec1fa1fbb1195dcfbce6b73fc2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09535951912403107], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c3173ace29bab9f36252d3edccc237d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.006920989602804184], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_693936cd82a19808679bb73ac3938130(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4420851171016693], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dbe9d282effd53c9f7db2929fc278574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.46214959025382996], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_2693c84dacec3b762da2332d1af8142d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2693c84dacec3b762da2332d1af8142d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ad385f0e1afd8b21794c89f5ffa8c60e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_07e314c5e77df14a5d8605177aa0d2b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ef356b968326a1e1685138108044f455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4667, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_870875c33d51dc9741b27759431e3472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([4667, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e2e14fa078024660b6b71ba7a5f19a93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4667, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e2e14fa078024660b6b71ba7a5f19a93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4667, 4], dtype='int64'),
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


class TestPrimitiveOp_2266d8fe50b5dbe351da3ac878770303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(104.07615661621094, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c653c3098864c2f4980b9a8e55a8f411(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(310.1311950683594, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e83a44b216cb7ffacb4e3ec4e64fa01c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e83a44b216cb7ffacb4e3ec4e64fa01c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fa2b1752e6ab14008fcf74c0e3145138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_605721da61f7fd33f7598ca4736d38e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1912d1a350180337a8370f341f520a83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1052, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_567ee429fe282d433783ab653929d76d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1052, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_61db17e4643daa3593dad3ce7008d0b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1052, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_61db17e4643daa3593dad3ce7008d0b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1052, 4], dtype='int64'),
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


class TestPrimitiveOp_bd88546d121092cfa151e48ba5466e1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(114.27332305908203, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7fe5b08b741699315acb1f0fd0b5826b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(5.179495334625244, dtype='float32').reshape([]),
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


class TestPrimitiveOp_36398c4d71083ca46439e6985a4318b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.05804389715194702], [0.03965532407164574], [0.025474827736616135], [0.08548954129219055], [-0.16984909772872925]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9c0aaae8f412b394653329381f897106(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.012821821495890617], [0.006005924195051193], [0.05297670140862465], [0.0015494964318349957], [-0.06175331026315689]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d3f91c97de81f7f9871b1dfc57cb0c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[-5.526961803436279], [5.602701187133789], [-0.5191314816474915], [54.172462463378906], [1.7504452466964722]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e2f3f96c0d42c8d64c78bc8d54de3abb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[6.526961803436279], [-4.602701187133789], [1.5191314220428467], [-53.172462463378906], [-0.7504452466964722]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_0fd174acf117a3175f93b0e5121ea51d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.035399943590164185], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_98b063fc2b060e6fb4edf2728adc6c21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4244700074195862], dtype='float32').reshape([1]),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3547fd270637d7be35e4452e30c8c07b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05550212785601616], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_a08bf1c93a8c69008e6cf8f1d5741a56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a08bf1c93a8c69008e6cf8f1d5741a56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fcc620c03875b6bd3d963440e720a7f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dcbe0605475738e379bf48024bd2f915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fe3ec13a1aebc0fc261bfdc9e14d3a6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2378, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3979e420e37309824a749227e6820f14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2378, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2797e433ca648dc6b3132189572f1522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2378, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2797e433ca648dc6b3132189572f1522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2378, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2113aae3463a81cd13ba4f69a8f7b985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2113aae3463a81cd13ba4f69a8f7b985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ae4fc4e9fbcd23a6c97b8f8c5902c0e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4063b116cd477832441b294f0d466b9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_62167ef663624604393cde11c1dd2bc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3105, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8c5992cab0b3ccad84d3445f17a279fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([3105, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_de7a7d2518100e80a8aa6c2cf12b0ff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3105, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_de7a7d2518100e80a8aa6c2cf12b0ff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3105, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_61a0c6d302c76eef268c92d2e741f3ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_61a0c6d302c76eef268c92d2e741f3ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b5b83ae48bae55f5c397f306e31ba648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dc8c5fbda28be7c922181e58e3ef5dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_789d4eff85be190ce66b96f49dde3785(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3832, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a94a85271b6759da1f909d9bd343c60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_55ba0c10f2e51154b62290414eca8463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3832, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_55ba0c10f2e51154b62290414eca8463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3832, 4], dtype='int64'),
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


class TestPrimitiveOp_81606f8fb7c3829b1e8adefb4a631a6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_044ca33194a1ad4c4c747ca504e49bdf
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.07457245886325836]]], [[[0.10637011379003525]]], [[[0.13256524503231049]]], [[[0.6309559345245361]]], [[[0.6628889441490173]]], [[[0.37623223662376404]]], [[[0.915749192237854]]], [[[0.6953400373458862]]], [[[0.4058386981487274]]], [[[0.13009586930274963]]], [[[0.32683053612709045]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_95064d459cb490134512f4976977d4fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([1.95993173122406, 2.0020086765289307, 2.203333854675293, 1.9218156337738037, 1.974346399307251, 1.935728669166565, 2.0282142162323, 2.100264072418213, 2.016552209854126, 2.1564738750457764, 2.1562886238098145, 1.99239981174469, 2.0882511138916016, 2.226702928543091, 2.169847249984741, 1.9368196725845337, 2.1782164573669434, 2.0199437141418457, 1.994685411453247, 2.0622143745422363], dtype='float32').reshape([20]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_205d7f7b93bdfda2625ab52b5269206d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(2.9692575931549072, dtype='float32').reshape([]),
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


class TestPrimitiveOp_99e54602192582e0c5ba50b403d8a77b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(358.12188720703125, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e9674b450ebef652ef701e9368ea81a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0026173554360866547], [-0.005801578052341938], [0.0918298214673996], [0.14991185069084167]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4129f8346ffb34121786ed418d36639b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.015165979042649269], [-0.0033329275902360678], [0.0007492093136534095], [0.0471089668571949]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eb32ad85f825961ea7b85b5f6dcdcb9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.1725807189941406], [0.7406852841377258], [121.56896209716797], [2.1822361946105957]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0269b92bfe8cea594a6a76024b50c945(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1725807189941406], [0.25931471586227417], [-120.56896209716797], [-1.1822361946105957]], dtype='float32').reshape([4, 1]),
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


class TestPrimitiveOp_74e27456ced8962ecd25ab9944ee7b23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(31.792827606201172, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e7111d55f5685f6096354527ba770245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e7111d55f5685f6096354527ba770245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0eb0e770ecca1315d4d895fc957c882e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3227124308c498c3eb9e78526f7018c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8d29ed9844708003a877025227245b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2087, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5807d3a2abc0e7874f39608d3469d8e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2087, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f8ed29194c9dad5c828691e313bc0720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2087, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f8ed29194c9dad5c828691e313bc0720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2087, 4], dtype='int64'),
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


class TestPrimitiveOp_87009b02a7aca42b72eb4317ee66f156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(157.8248748779297, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_46515a6be03ba2e25580e431ac461d09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(2.636561870574951, dtype='float32').reshape([]),
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


class TestPrimitiveOp_23d658a871fa8271c0a24131fe7d137c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(35.19792175292969, dtype='float32').reshape([]),
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


class TestPrimitiveOp_1e951d5600d0e936aa83cd081d90c11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1e951d5600d0e936aa83cd081d90c11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e13f63835af8de264d15058a8140111d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a60cbc6d2716dd5fa6f3bb1ae6508fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3e83dc66d05dd09f756b1066059fbd7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4271, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_11591a576cfa6f254ee8af08567dfbe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([4271, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5b299689c8b5a76a315b5773074950fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4271, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5b299689c8b5a76a315b5773074950fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4271, 4], dtype='int64'),
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


class TestPrimitiveOp_97bff7ad26affca6c10eae21dc68ec6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(80.11605834960938, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3a60112ee8cfe96a66a5a61be5709c38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(4.487279415130615, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e427bf2e4c853cc32a2ecfae7ba4b217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(35.296112060546875, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()