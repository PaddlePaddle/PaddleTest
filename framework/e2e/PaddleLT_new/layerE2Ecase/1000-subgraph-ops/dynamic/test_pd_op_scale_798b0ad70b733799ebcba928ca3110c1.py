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


class TestPrimitiveOp_b2e89b9c40776c81083fc8811ded2d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_c7dfb26f591072bd521d3facdb763200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0114a7e611aa0911ce0aa68e78e2beeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_3be6e43d9724a14a43b3692c3adebe67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.33082929253578186], [-0.12685734033584595], [-0.08850478380918503], [-0.04121403023600578], [-0.022963346913456917], [0.014477982185781002]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_20e6464066579396ff6be6282a91dab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0005391706945374608], [-0.29754284024238586], [-0.046392522752285004], [-0.05150821805000305], [-0.004321814514696598], [0.08872191607952118]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_2e8c74266fd180aad2b552875ebd37ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-6.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8b0ba424eac094ff1886fe6aa530bf0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.27302616834640503], [-0.14556801319122314], [-0.12867471575737], [-0.20281419157981873], [-0.20082539319992065], [0.1963663101196289]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([0.08333329856395721], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_435182318148bee7ddd300f35cf68b06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.022752171382308006], [-0.01213066279888153], [-0.010722888633608818], [-0.016901176422834396], [-0.016735441982746124], [0.016363851726055145]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([6.28318977355957], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3e4568cb9f0002fc77c2f71f590879f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007487208e4091dcd686a62928d8f388
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.1429562121629715], [-0.07621925324201584], [-0.06737394630908966], [-0.10619329661130905], [-0.1051519587635994], [0.10281718522310257]]], dtype='float32').reshape([1, 6, 1]),
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


class TestPrimitiveOp_f60d78e4d19e5f19fd199ac3df046071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_11e9b6ebdb51097b66170f7689946c4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.75], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_4872b9a014d85b17ee54da032f63935e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_ead20b4717dd614addf97a13b77cadd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc09fca21912eeca52f64e68a76e6d4d
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a5974194e782e9dd4db4483d866aa842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9cf76147382e1c6c0d8f6d4463f42fdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ead20b4717dd614addf97a13b77cadd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc09fca21912eeca52f64e68a76e6d4d
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a5974194e782e9dd4db4483d866aa842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4066b1156f65912e1f3c506880a71591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bae0f6cff1c49c7f97ac7d81d93c9632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_4374fb9ea4fef0de5433528718f5139d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4374fb9ea4fef0de5433528718f5139d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dbbc84b6d41bb94b1d846cf46b338000(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e239ba65dfb03499e870daf71e1e2dac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1524, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_162862760cf83e1f8998741f73340cbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1524, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_65c0b8522e5205c69f44225e94c0fb85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1524, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_710ec7ac755faa99912adbe421bf744d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1524, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_710ec7ac755faa99912adbe421bf744d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1524, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_66c569f6a28eb6188679de885d3945b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_66c569f6a28eb6188679de885d3945b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_14259d3f950d6b1e95c8a2261ea4f418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8d64c16cb6dc6378dc0dd65aa2f4c05c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([2340, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0ad1e4e7bb74a275d56bd12dbbbf12bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2340, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_48485d74db67fb0b5aafe277d440b968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2340, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fca89cb5493c8fd3bd663237038f3e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2340, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fca89cb5493c8fd3bd663237038f3e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2340, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9812155e789c91c7f5928885135cdc6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a134be30792c0efbf359c2f97e111922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d74718ce6e72449d548c2271c12ada1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_8d2627ec06551986ca6387b77b382a21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f4699da7c0237c8449b4859ecbd10a
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_de98c0066a0d447b3885f4c1c4e5cf57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a788f85c0ba37de2381a706ea5c30863(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eb1e729b0e99679aea779e2c1111f63e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab3150389d4d0f36f8754ce0223d169c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fe81ccf915b99d12689d6af269e05602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20638607442378998], [-0.0044953045435249805], [0.2725278437137604], [0.06655335426330566]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_12ebd0e2e3492d3c2d029be172f2c851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2855321168899536], [0.35548198223114014], [0.0614340715110302], [0.12111995369195938]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_edbc0df517696f0a3ea3157616c25579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.27718788385391235], [-1.0126456022262573], [3.4361026287078857], [-0.4505169987678528]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b40bab53010706010824d3b739a6dc34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.2771878242492676], [2.012645721435547], [-2.4361026287078857], [1.450516939163208]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_936dd0fe17599a225fe257ee32cc0aab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e66d92eb1d14b7649473890dcad8bf4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6d11ffa4f66dd6c8a9e405ed33972835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0958900451660156], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5c1712e550192dafe9e01dcfec82097d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.0986913442611694]], [[0.9473166465759277]], [[0.9024375081062317]], [[1.2308518886566162]], [[1.022817611694336]], [[1.4777599573135376]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_98dbc7d9c72da48b53725b3eb18a9de8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.1539826393127441]], [[1.0024223327636719]], [[1.4130655527114868]], [[0.6176849007606506]], [[0.7467498779296875]], [[1.4021321535110474]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_0dd5ddc400c1b070d1d1cf5a94a23baa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ef2f427d2d78f4556efb1a112c5906e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_9229146a42c122787c34da0675d9b6d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06ff4047dd00b36841984b859d27834c
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bb448f87a375e079d7b57da4d7bae68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.1940300464630127], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9111cf4ab25d53e51c7039dd8a56d3bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76d9c535a1587afcdfe14cf6ec35704d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4289543628692627, 0.2477843165397644, 0.08632159233093262, -0.6113117933273315, 0.24053460359573364, 0.4474496841430664], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_521fc3ebf478b5c4f1cba1de9a0ca825(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.01740086078643799, -0.17055979371070862, -0.4457579553127289, -0.19193696975708008, -0.6420028805732727, 0.4727821350097656], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_36d2d061ace591336f5c4e4d6450df6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.011115729808807373, -0.00794142484664917, -0.8344384431838989, 0.04834216833114624, -0.36716797947883606, -0.32251909375190735], dtype='float32').reshape([6]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8ccfd1d5c3a9e53d5316b742acf20bb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.1623883843421936, -0.3969222605228424, 0.2646040618419647, 0.04656553268432617, -0.005823075771331787, -0.33780205249786377], dtype='float32').reshape([6]),
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


class TestPrimitiveOp_a84dac78bba6a4ccd083bba02efa9e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6e893b1e74b938d469fd1956bd5fe0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14890655875205994, -0.49899178743362427, 0.0836489126086235, 0.07322786748409271, 0.09691885113716125, -0.142303004860878], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4388ee627dc18d7b02624d9179c9ee34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6e893b1e74b938d469fd1956bd5fe0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04890262708067894, 0.029158905148506165, 0.3381032943725586, 0.12300670146942139, 0.19350680708885193, 0.31247466802597046], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0681e0b3241382d4f635d4306c28e002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6e893b1e74b938d469fd1956bd5fe0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5902164578437805, 0.2567071318626404, 0.17809712886810303, 0.5459249019622803, 0.11352415382862091, 0.7081145644187927], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5e119718aae59ebe4eec87332ceb4dab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.5218230485916138, -0.7459564208984375, 0.09630866348743439, 0.6822724342346191, -0.9187926650047302, 0.04123502969741821], dtype='float32').reshape([6]),
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


class TestPrimitiveOp_b69f78132972afc8ba319f9b6cd5e6cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33fc97b08736fb3dd9b6dba9d150c721
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.0, 0.0, 0.0, -0.0, 0.0, -0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8a09777eb12b7e8836331aedbe7809f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6e893b1e74b938d469fd1956bd5fe0
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9386179447174072, 1.225521206855774, 1.0037591457366943, 1.1886584758758545, 1.3421335220336914, 1.000689148902893], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a2d87b46e6360c416b4e4016328ba37e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([1.5373048782348633, 1.1550887823104858, 2.89843487739563, 1.255260944366455, 2.7917590141296387, 1.4412775039672852], dtype='float32').reshape([6]),
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


class TestPrimitiveOp_4a6678d1252c2e4b4bc9818466e129ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(1.846521019935608, dtype='float32').reshape([]),
            paddle.to_tensor([10.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f00d9d9386c6c3f75cbb0a9776d89f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(62.326622009277344, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_72bb5299ce6e09b77a4392bb3cd34ea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3a72feda562820470d825f7fc21477fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fdbda7f418940621c9ce242781a66cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6356da22741102d24dc18b708fe97d04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.01010000705719], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_00fde3a2fe3c97f2a79b5534d94fffe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc282ce2c0868b98850e2a9bf20e6071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_24cbacf5cc3bd3848339d33ff3194c86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f83fad2d70bd75524434d8e42101d8e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ae15d9e4db29685fe17a0008415a7d08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4bcbc9070fac93fee903642ecfbdcb9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_208d189f5c7e15a3d14ab29f9f09906e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_25d132edd3d0fc0a7507b9125c252971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.75], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7ccfb723ac1a16a5565a6a87137fd0f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d126e3818c07e637124a4e1f0fd065ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc09fca21912eeca52f64e68a76e6d4d
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09d70075425c410b40da045a183153ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc8467493478808bdfaeb2befe4d250e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d126e3818c07e637124a4e1f0fd065ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc09fca21912eeca52f64e68a76e6d4d
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09d70075425c410b40da045a183153ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bed9bd8913ed25c2c696a31cb80a5f62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_378550e72c1eaf27b1b168a75e3b54ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_876a8dee2aeb5aabe91b68fa87d0d05d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_522de5903755e6b30c5eb7655e2fc895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7bbed166bf1261660827a294d91238da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06998768448829651], [-0.04479404166340828], [-0.366666704416275], [-0.3071730136871338], [-0.10413970053195953], [-0.00407062005251646]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6bf44cc62631cecc06f2d3e672936c85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06657133996486664], [0.012212522327899933], [0.09976665675640106], [0.0372019037604332], [0.18982329964637756], [0.01324944756925106]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cbf7194e2dc6761a1bbc04472a5c75af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05131854861974716], [-4.667878150939941], [-4.675242900848389], [-9.256916999816895], [-1.5486140251159668], [-1.3072293996810913]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d19016bbabbd276a992e84bcd76400d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.9486814737319946], [5.667878150939941], [5.675242900848389], [10.256916999816895], [2.548614025115967], [2.307229518890381]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c4f7689024965c539f296d9cc7e8879d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6356da22741102d24dc18b708fe97d04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.01010000705719], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d8a64d2820569d48282ec8dfce2640fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1100cfbef9833f8825df97a1e1d11d48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_151fb6bf1f94dab4728010cda567b378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_2d0d731a228ceb0a7ebf24195ef1540c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_079b021238f4ac5824a617c7f23bc5cd
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8122660c5a068922a6df9bb663675664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([2.3151187896728516, 2.07491397857666, 2.024224281311035, 1.8722412586212158], dtype='float32').reshape([4]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bc2eca0d5de97d422c1b5b3ed0ecc72e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(-0.6158589124679565, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_91ae4ddeaf1d0ecb61bae8dfe31d7928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.uniform([], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.09090910106897354], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b3e033906bac36600bc3b723d217c555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.uniform([], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bff2e9f0980b98f8d3e8c1ec4976a445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.38789671659469604], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_0da5ec97c595aa2d7c3935fc34441e2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bd00524ac898769d2f7398c510382e4
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5fb8f7a7c7d783efa53bc3f72dcd7020(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0526299476623535], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0bed290b74bc4ddeb406fee5f841732b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


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


class TestPrimitiveOp_ac94571afded5fd5b98eb23c76819deb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7e9868b5ab8ecf307dd1ab5e07d57a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1100cfbef9833f8825df97a1e1d11d48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f8e63ff5f1301866facf2818a0e1cf7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_164925e3f1aa62210eedde47bb20ff77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_670b74f83f35c1414654065e8ed0d907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2ec5d225064a5dfae9eadcd78184950d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_1992e4bfc71f5b28706de7722f02dba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aba2c2b79f05f6aa92c3e6461da8f0f
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1992e4bfc71f5b28706de7722f02dba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aba2c2b79f05f6aa92c3e6461da8f0f
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_329f7d77b8b6a65b51cc2d01281bd8fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2b6f333b37f7930d18d4e6b58bcab6c
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_329f7d77b8b6a65b51cc2d01281bd8fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2b6f333b37f7930d18d4e6b58bcab6c
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_188da53b9999560b81ba350de3d516cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8e04abc4133ade1bd13d15ffff2c1525(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_574b778d77bc75ac30a5c48f895674cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_83c89bf0e65b78f6dbc51199ee1ac513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f8069cd77b6e63ce8f81853f0fae4c95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_008c028cb7e50589906d568dd2a77349
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f8069cd77b6e63ce8f81853f0fae4c95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_008c028cb7e50589906d568dd2a77349
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_2ddbd3c9b7a9eb71d524e95b01e745e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7488700375ae67f959415fc79a3f112c
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2ddbd3c9b7a9eb71d524e95b01e745e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7488700375ae67f959415fc79a3f112c
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_412c64110b6ca5d53a8bdefb8ce66150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9bcd06abfab0a0c8fe782e0428600376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a1905f51ab24c8e8547d6893471c080b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37776f80c503717897db4af0a8c16006
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a1905f51ab24c8e8547d6893471c080b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37776f80c503717897db4af0a8c16006
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_697da294005aa1553663b3cae1d2361f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cc86a8c6809b87f325d839d78737131
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_697da294005aa1553663b3cae1d2361f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cc86a8c6809b87f325d839d78737131
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_ea70d61022e908ec862947e8710f7e80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d97f9438fbd1839f4e4f1c8d8047477
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ea70d61022e908ec862947e8710f7e80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d97f9438fbd1839f4e4f1c8d8047477
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_c1014adcdb430f0ab483bcabed4e2a96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67271733891d5bd01ade7457bafc00cc
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c1014adcdb430f0ab483bcabed4e2a96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67271733891d5bd01ade7457bafc00cc
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_9263f7ea7acbd0947b8fd6930231aa7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c53c097ba6b917cbd2d8002a00fb7
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9263f7ea7acbd0947b8fd6930231aa7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483c53c097ba6b917cbd2d8002a00fb7
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_1c8909b4df3782644e40c34da149f8ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91dac0b3ef651da0145a53c05f97b419
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1c8909b4df3782644e40c34da149f8ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91dac0b3ef651da0145a53c05f97b419
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76ad9f60e790eb5c470f0b30bab39f7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_83d2449b2802c29660e89a9d86b6d14a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92cb719a58e611bb95c2eeb1f148340f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.7520722150802612]]], [[[0.6359053254127502]]], [[[0.39528268575668335]]], [[[0.9229522347450256]]], [[[0.9134716987609863]]], [[[0.34799909591674805]]], [[[0.0673963725566864]]], [[[0.9767506718635559]]], [[[0.5640743970870972]]], [[[0.3461367189884186]]], [[[0.7982662320137024]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4cdb2d7e4ae711d77635f29cec49edf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.1764700412750244], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a6cb1c7f10bbb15218dc8564d59d724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_a9db54db8c37f5e40f82d226bf67f1c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc5dc1a9beaedf06881268b87297646
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e790f9bab65d0a224c0147e709be2ce1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2a6cb1c7f10bbb15218dc8564d59d724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_b06b5fee566a7added2ec41c23556c7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e699dbcea2e3a941b73d9135c26a1f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1fba7a62a2f37f7b3798f028cbc9aa41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0256400108337402], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d33b76ab6839e20dab60f1d79b6b85b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7a34e8faa1a4b59eb060067c9ddab7da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7a34e8faa1a4b59eb060067c9ddab7da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_79e4e7a75e34e1a936e47c34f3a6ecb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6b59c11629d2467d2d051ba570b90437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_38a12d37ac888739ef0c497c5580d32d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76a2e5dd5f16ba2ea1231e3e32d5390e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_8c9f2b6daa86cd7f4f29833de22375d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18c64c64a19beef4e5afcb80a70fb2b
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ec26ed6054a536bf0d1eee34001a357d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.1428600549697876], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e4eb2c9af5ff9a3e252d1174095e277d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.13141000270843506], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97a6f85a69a2e021ecc1b07a63409cf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.2926388680934906], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0fe16d28dcd79edcfd4cef9f7c85b93f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.1643106937408447], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_169a4c7942abed0754788007ce3190b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3a5a0c4f8628ee16948a7d76eec21ecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c6020423a518cba065f78028469b780e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_38acc7ee66000c8be3358b9d77cfa02a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_38acc7ee66000c8be3358b9d77cfa02a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3d21114834b3f4aeecb893f04f0771b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a1f01a5c04caa3cc5b9dcde2da9cdbca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([2047, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3af4d41553b9e2d7f423980e30ca0102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2047, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f57a97d2d7947b224d03effd8d6c5cb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2047, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0e8d21e52e198c8d7176e45ad93c5e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2047, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0e8d21e52e198c8d7176e45ad93c5e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2047, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aed7ee53572966dc4636d534e145d052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cfb076b3a21282b4c87ec10e2d98700e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6847228e201d1f65cc544eaa8a398901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b1ef7d2902395def691798066f236d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_044ca33194a1ad4c4c747ca504e49bdf
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.25073614716529846]]], [[[0.009872601367533207]]], [[[0.289939284324646]]], [[[0.463524729013443]]], [[[0.5436007976531982]]], [[[0.40612366795539856]]], [[[0.9474888443946838]]], [[[0.632882297039032]]], [[[0.11343371123075485]]], [[[0.7267495393753052]]], [[[0.5690246224403381]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7b92f671a9231412e0ee6665fd9e1473(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0810799598693848], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_84be48ce31cc54fe28e76e59f71fa2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_644ea3e11ec07350eed6eecbba64998a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_623f6e8ba8cc819e5b35734aff742513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(162.770263671875, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c9a167c231f37062cb7564994db57274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(186.10382080078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6538848c9218756772da6597a4d5b4cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e4d18e49a0eaadd8c93bf2fbca4a574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4848df135b982002147361228ea47469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b8abf1bb3fc4adab04a7cf0b9e3be652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_47acde84d3595be23a5fb64fda78f1b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c7d7fa68330c54a366a91c48d2bce904(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f59f893beed389c4ee8ae999888c64af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_23fa1bd5f11997ff67f730f17d1f3655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_490d263326fef7245195bde15bf51d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_92493ffc55eaffeb74ad2f16f2f884d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-50.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ebda7cc7c4efdf10d9873f5c5844d33c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc5dc1a9beaedf06881268b87297646
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_97f4205b8f4c5ef9bf3d7bd4ab87cd33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ad901329f060790fafa50461e9c1a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d504efab29947245201d7a36161b2615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c60965428468839b471c8e6865284a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_54ab11aa1514eb34e8f74f0efc2b70a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_15308408bce74487fa88e7b95b284878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ad901329f060790fafa50461e9c1a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7c0b84d840ca6ac0b2b2a36119525fcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c60965428468839b471c8e6865284a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_9ce8f5a1a602cd4c2bc0b37c996173a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_079b021238f4ac5824a617c7f23bc5cd
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b19b6d6a1079218e32b4be5a8e0db2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([1.6792073249816895, 1.937551736831665, 2.7713794708251953, 2.211115598678589, 2.4154906272888184, 2.5097436904907227, 2.1516122817993164, 2.4055728912353516, 2.623506784439087, 2.1877293586730957, 2.411167621612549, 2.255079746246338, 2.168220043182373, 1.7224218845367432, 2.2600595951080322, 2.1672592163085938, 1.985633134841919, 2.344369411468506, 2.098412036895752, 2.5383365154266357], dtype='float32').reshape([20]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f1246bcb8b194f26243fbfd9f60c5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(0.026168107986450195, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2e4f99c27cbd0014ea12d08941f1a569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.37008610367774963], [-0.20031902194023132], [0.09816905111074448], [-0.1653597354888916], [-0.2358454167842865]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_010c72dfd6e3dc24939562bc9d1068e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.12076050043106079], [-0.24472537636756897], [0.7353484034538269], [0.1291397511959076], [-0.006730019114911556]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c5c63304a36f4d9f78f6059c2a8828ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.064628839492798], [-0.18145382404327393], [-0.8664999604225159], [-2.2804713249206543], [34.04379653930664]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d0120d1255821ca35f965091bf4130c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.0646288394927979], [1.181453824043274], [1.866499900817871], [3.2804713249206543], [-33.04379653930664]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5f6f00740b0cda3f5c8a2f0c77483741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5f6f00740b0cda3f5c8a2f0c77483741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c4047a836d96fe63dde126123c5691f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_08ae9b5ad13a23988d518adafa437985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1813, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_33eef8168e463f59ff95c444861b3cd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1813, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_96dcc43d5c20be6efab46ee6cdd6e04c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1813, 4], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_277b722b33fd2f672744a21e9839f2b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96a0c65442ab89e3c53601857861ab7d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1813, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_277b722b33fd2f672744a21e9839f2b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96a0c65442ab89e3c53601857861ab7d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1813, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_38bc4529f45bb993279ac68fca206cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_adc047487445a88a309c362e834a9b44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c2af6f9fae9ad9af58d6f31ced9ab1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_55cc184a027da29ca4863d00f67a9d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_079b021238f4ac5824a617c7f23bc5cd
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09c338611a89a682b84dbb935cfdd605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([2.4750375747680664, 2.1069557666778564, 2.4038169384002686, 1.8408586978912354, 1.873691439628601, 2.522643804550171, 1.9634921550750732, 2.4064550399780273, 1.289695143699646, 1.6701390743255615, 1.7007286548614502, 1.9724042415618896, 1.7032008171081543, 2.227449655532837, 2.3570191860198975, 2.1678831577301025], dtype='float32').reshape([16]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6992bcb758df7fccc414e478a522444f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(-0.6256986856460571, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1fee455bde2c4f29ec14b0eeb7ca99dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([4875], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1fee455bde2c4f29ec14b0eeb7ca99dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([4875], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7ee231c8a0690ae8752908be0d371d9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(412.2397155761719, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9a2f02cc4b8639e37e60def79613dd16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44393253326416016], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_005185444c936886438fa152abea070c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b7ec0037f0cbe943f59047d875f36e7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03575217728130431], dtype='float64').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1e268c3f3b585b449445ddb781a23426(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_3f1a6e288db325c55e2821fa5e6b8017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0f4f967123d907137366e0b4bf855ba
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3f1a6e288db325c55e2821fa5e6b8017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0f4f967123d907137366e0b4bf855ba
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_0ca93aef354b26262e09f1e5c045cf61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fa3b36bb5e62b6cd0445bd195358538
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0ca93aef354b26262e09f1e5c045cf61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fa3b36bb5e62b6cd0445bd195358538
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b31d8a677434b4a1403620cf08459d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3876c0453f51447ca5e6fc43977fd8f4
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b31d8a677434b4a1403620cf08459d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3876c0453f51447ca5e6fc43977fd8f4
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_28fc7ba71e0650e35317e81128c268d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33ecc32083c24d6f8a3157023550e645
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_28fc7ba71e0650e35317e81128c268d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33ecc32083c24d6f8a3157023550e645
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ed6c4a298e92e6d00772468189c96ebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5315b45a4745aaa82a93f928fc89f0ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a944f0e66c033cf0bb4f43d9b5b5dabb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85e145a982a34a7b9e627665de8450d0
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a944f0e66c033cf0bb4f43d9b5b5dabb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85e145a982a34a7b9e627665de8450d0
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_c8ed686467557e8a478ab68169a449ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8618280eb7f319e14c1f08caca4dc14f
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c8ed686467557e8a478ab68169a449ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8618280eb7f319e14c1f08caca4dc14f
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_725f1d7274fa37360344fed806da6fe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_725f1d7274fa37360344fed806da6fe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e8fdec172466dec14b2f59c6b33ef71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2ad7c2a7a1a62826b2b735721d544821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([3061, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_63f6ee493122afc09c7235eadf697d81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3061, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bc0f788fda974a9b52cca0879b06f815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([3061, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7b9a270bdf1ebde47313ad6191fb7b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3061, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7b9a270bdf1ebde47313ad6191fb7b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3061, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_367be3469dea6c6b1a20c5774f755539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5d0abd2941a972019d8d51702c9a00b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0be24031172d340645384488947aac0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1d7d3b8bb069f9b68f9ca177c3dc27e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d0d848d338055fcb48bf612e92086e62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_36056bcbf002e0400e83b6d67eea5df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0183dd53258a8307c6217fdedd344a19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc5dc1a9beaedf06881268b87297646
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_db97e1875cb4937dd35e3e0d61bccab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_36056bcbf002e0400e83b6d67eea5df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ec389a143b2ba6cafaed0b4fdfb5c581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ea66575734bd05b07acccdfff51216f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cba1f83dd588da787529b6c4f2942a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc5dc1a9beaedf06881268b87297646
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f220a4cf004bc302850a0efe63dd7de4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ea66575734bd05b07acccdfff51216f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3102bc0724fae555e66b5ad854584874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0041218101978302], [0.02485261857509613], [0.4107665717601776], [-0.22349874675273895], [0.1249942034482956], [-0.012953147292137146], [0.46127960085868835], [-0.0874498188495636], [-0.32099252939224243]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9c52e6bb6af0cd68971f4b5eb731b015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16105230152606964], [-0.014222315512597561], [-0.05521157383918762], [0.040287330746650696], [0.02169066108763218], [0.2522542476654053], [0.39104196429252625], [0.059318866580724716], [-0.06997423619031906]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c2c07465c4ea69b59840ee6ec42d86ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.9744070172309875], [-2.747438430786133], [-8.439863204956055], [-6.547618865966797], [4.762581825256348], [-1.0513496398925781], [0.17961661517620087], [-2.4742329120635986], [3.5872957706451416]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5e6d7f59a7572df81813d0ef818475f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.9744069576263428], [3.747438430786133], [9.439863204956055], [7.547618865966797], [-3.7625818252563477], [2.051349639892578], [0.8203833699226379], [3.4742329120635986], [-2.5872957706451416]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bbb08948e4cbe937a84bbf7efd5bd5eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_206b6a9d60f3d20e399215a6045e5da1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc5dc1a9beaedf06881268b87297646
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8511770ef791c1014c54b4b8e9337600(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bbb08948e4cbe937a84bbf7efd5bd5eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_21abef76d846d3041dad364bb2164404(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.717670738697052]], [[1.5980790853500366]], [[0.6110512018203735]], [[1.356167197227478]], [[1.23806631565094]], [[0.743411660194397]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f2dde5c2086c06ded015c2badc2ce871(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.9611691832542419]], [[0.9671278595924377]], [[0.9033939838409424]], [[1.3020422458648682]], [[0.666857898235321]], [[0.9617844223976135]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e13064d5c486184e8b61b72108856cdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bd00524ac898769d2f7398c510382e4
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.40162405371665955]]], [[[0.895107090473175]]], [[[0.8396212458610535]]], [[[0.4200904965400696]]], [[[0.2215631753206253]]], [[[0.13276486098766327]]], [[[0.6387240886688232]]], [[[0.8759875297546387]]], [[[0.5385084748268127]]], [[[0.6328368782997131]]], [[[0.957517147064209]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d91a4ad4e108db767c2c2a87db70b7c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0526299476623535], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c7b37ef4e1f297dd941ecc0223926f4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4d7d40a032e9f2c96d0146f07bbf7099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4d7d40a032e9f2c96d0146f07bbf7099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b9f1cf208334e4ba68b02b7e98496aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_330f6dd9387cec1858aa836d5189a40c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([2062, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cb0ec98d6565238a5090260d494210c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2062, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5b2f5b634d3d99fa2f0670fd233f2959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([2062, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_567b767233bd1d917b8e89922b6751e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2062, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_567b767233bd1d917b8e89922b6751e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2062, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e4d18e49a0eaadd8c93bf2fbca4a574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_498f881652259e4420026cc3be63329e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_099b9c000653023483f390cddbdeabda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bed9bd8913ed25c2c696a31cb80a5f62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2e938bf4bdee64a28e273af829607650(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2099c68d21d771684d4810694692060b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9a6da9e0f0cc1c5ee706481c879e839f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c412410589329f5bf9da4b4c4e54a3b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d57484f92358c6caefb611afac796b16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_abd9986199d4cf1f8128dd217d98132e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ca1a00224cdbba8ec4bcf1cca2663c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d2596ac14513230c04a40d1f96ba4c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e2f96773d5ec941d983e8ca3038fc409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.09946253895759583], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5ebe9b244465e81da2ce2808c6400c89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.46824562549591064], dtype='float32').reshape([1]),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0111004361b3feeb507eba9aac110b2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.035158753395080566], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ecb74b1a0db33dec5f1034260d90fa68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7fe924e359d088bd549f84404b6c32c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d007e1b5394500bb78f0bcc2eb1dc999(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_15e3c044ac6813854473635f819bd505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2ce70d5cf6894e5b36c15cee0885e38e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a4ba897926ba32ed1db3ae4bbb98252c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b862d3f12b36562219b6b4babbfe41e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8e23872067ee417dc74f3f676a0350b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58f208baa1620b964905af37f559414e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eaa5b24af245e0260d218c215509f9e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6601bc42a9c3f7be2745af736c17c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f572c295acf92b2f755359bb0aec223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_926bc6237f6cdd8946f3837a7b880e9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e159018cb953531937774a04000871d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_602cb6581f328819b65c5d08d81d9167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d3c05274f238e7e5a83f3e7343884bff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_03b5fcc4cbbd1f97f884f73a80767a26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6c66603acccb66e8bec3fc1b7596b867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_37e27abcebef7ccb1a18a3dc596c656f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5578afdfe18ea60c93999e907f0f90b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d3c05274f238e7e5a83f3e7343884bff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e8119d87e3de5c4c68ef91021f50968d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6c66603acccb66e8bec3fc1b7596b867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_38fe2e6f37accf491bd37b9272955d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_caf7564bc867afa46267be8136ab906b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5c02284a5c3420159cbe089784bae093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a5dfad317907f8be48d45e3202085138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58f208baa1620b964905af37f559414e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eaa5b24af245e0260d218c215509f9e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6601bc42a9c3f7be2745af736c17c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f572c295acf92b2f755359bb0aec223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_926bc6237f6cdd8946f3837a7b880e9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e159018cb953531937774a04000871d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_06f70935e9d9c3936a97cd6dc3c10883(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3183362809672ac65c0e3a965edfbce9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6c3111541f933c33540d9526f8bee23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_33e4d67c78b66ea52b2e13950a8ec2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f815c3a40e555f7fc01b936f72964f24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cf4fa7be64cd89216457dceefa2d5107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f4699da7c0237c8449b4859ecbd10a
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cc266b1096c9a60bd3520c06dfbf2b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2595764696598053]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_92873ad5b3653fd5556dfc31935a7d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0166384968906641]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c4b650b7d60b56115ff9eae68a1d11e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.to_tensor([[-16.600955963134766]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e4df907157deb62b88b4420f3326a576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.to_tensor([[17.600955963134766]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9111cf4ab25d53e51c7039dd8a56d3bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9111cf4ab25d53e51c7039dd8a56d3bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9111cf4ab25d53e51c7039dd8a56d3bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ba230fbccb8913df9ec79a36a76e4489(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_106ecd70fddbc2b099cece6e9234fef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_50304fd42e92eb17770aa11077105f23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7833fc520390b54cd52cd65904a05a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bb58ed9e902c8a3a5367b615b9fb68a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aeb8c54335db673dd4fb895605039007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_10dd94a84e20b267aaebcdb7b96ba517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a5bafb49d79e2c8409c8166edaae71c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7833fc520390b54cd52cd65904a05a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9ca93c13dfce8f1607453bd715d4e2b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aeb8c54335db673dd4fb895605039007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d50aa900e8e6f47a5851f600fc11d432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(76.51245880126953, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_06f70935e9d9c3936a97cd6dc3c10883(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6e509b4f670ab31626ae40c22a1b21bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1f24c5b8ac616464a690b299f5fc4809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_4bbcd3c8135faa6ae51f1b467207c2dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1002], dtype='int32'),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4dfe8751a52f8c511365f98f0d3febf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1002], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_03ed035a203f58e593de55105f71c356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_38fe2e6f37accf491bd37b9272955d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_38fe2e6f37accf491bd37b9272955d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_38fe2e6f37accf491bd37b9272955d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4d48fa58fd0082d69b4cb87cccac04f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aed7ee53572966dc4636d534e145d052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_172d8abd175640353b6e5ba5203ab41e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7b9ab21ba0293e5e1427851a5b4d24e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0f2145aa82bf7cf6a45231ec518841f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2e89b9c40776c81083fc8811ded2d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9d45d99764855e63a31ee3682608954a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09bb6804146542f57b9a14eb0b7f892b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9a571aecba9a17b8992858770cbe1937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9a571aecba9a17b8992858770cbe1937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_80df6db1ad7f3dd7828f54528c95a063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2fa7166c91d6532cbe0c3bbe15fd41c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([5526, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8a2124171b43a7188e198ab6d672c6a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5526, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7a4589a337c68d4014182612c7f977ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([5526, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7c30461a4759c916386f3700156a3618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5526, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7c30461a4759c916386f3700156a3618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5526, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8357acc5d51ed4ed39b504040ad6028f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_169a4c7942abed0754788007ce3190b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_88943ad6ac5158ae540455dd981e0b15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cde12c41dc191612a45ca8b8527d952b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_729d6cf1654c4afd133201df7b1b18f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e699dbcea2e3a941b73d9135c26a1f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8087913393974304]]], [[[0.40785327553749084]]], [[[0.5567630529403687]]], [[[0.6905555129051208]]], [[[0.7094281911849976]]], [[[0.2813236117362976]]], [[[0.21070612967014313]]], [[[0.36984360218048096]]], [[[0.965480387210846]]], [[[0.30857375264167786]]], [[[0.03557150065898895]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_46bae2e643c3b6ba7a57e5ef84d2e173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0256400108337402], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b78c860407722137f7e9b7248e4d29da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18c64c64a19beef4e5afcb80a70fb2b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9334134459495544]]], [[[0.22188784182071686]]], [[[0.8801380395889282]]], [[[0.48282238841056824]]], [[[0.16895535588264465]]], [[[0.5548867583274841]]], [[[0.6471811532974243]]], [[[0.165309876203537]]], [[[0.36543476581573486]]], [[[0.22294043004512787]]], [[[0.9333899617195129]]]], dtype='float32').reshape([11, 1, 1, 1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_330f6fa935b19e12eeb68a99c0205632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.1428600549697876], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_caf7564bc867afa46267be8136ab906b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1799e8ab80110d421108049bdb758216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a382db619933473712a489bf009817cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c012a3109e12169269cb93dacca2751d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c012a3109e12169269cb93dacca2751d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fa76239dd1e1be958ac9d97747d8b132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d64115030d850e2c0419bd6c384fc8dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1071, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_89d16dfc15a70fa04c8a012860ea468e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1071, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_480a9190fe873a516da21303c2ae20dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1071, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_310e92cf6631b33d62d0eb736e00db18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1071, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_310e92cf6631b33d62d0eb736e00db18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1071, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7982d214b4d58a981bc3c159d65b0e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(65.42365264892578, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c7b37ef4e1f297dd941ecc0223926f4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_643c13983f8cd4bcda70cc43c34ed468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58f208baa1620b964905af37f559414e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eaa5b24af245e0260d218c215509f9e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6601bc42a9c3f7be2745af736c17c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f572c295acf92b2f755359bb0aec223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_926bc6237f6cdd8946f3837a7b880e9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e159018cb953531937774a04000871d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d8c7966b3680f6694f5dd2150b551ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d8c7966b3680f6694f5dd2150b551ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_db6625c79397445a832e0cb4f2106706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2b4800366362387c8fe13e3a2032db8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4c111a46d603bcb56d49b1ffebfa4cc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1760, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59bb2cedc1b8bc5f3306a5a9b1ee29a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_df02a3cfe6c69ff1c1abf711ac5e4b93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1760, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_df02a3cfe6c69ff1c1abf711ac5e4b93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1760, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_31201d506387d570bd5ed22f201444ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_644ea3e11ec07350eed6eecbba64998a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f0a3933d8455ebad97dc39a671894bb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(193.79635620117188, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3b3fa9f6c88dd5c20bf4a027fc3cf5eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(112.9666519165039, dtype='float32').reshape([]),
            paddle.to_tensor([0.00390625], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_de98c0066a0d447b3885f4c1c4e5cf57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6fd5663c3f0e7639c7072e8a5b41366d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6a5af67e639ac311a5e805e5166f33df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58f208baa1620b964905af37f559414e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eaa5b24af245e0260d218c215509f9e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6601bc42a9c3f7be2745af736c17c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f572c295acf92b2f755359bb0aec223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_926bc6237f6cdd8946f3837a7b880e9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e159018cb953531937774a04000871d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_adc491b97e2455df31eb253f563a7a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_079b021238f4ac5824a617c7f23bc5cd
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c7c40197f7e01f0064c1b21100bd7493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9889304637908936, 1.8134485483169556, 2.136913299560547, 2.20802640914917, 2.1006884574890137, 1.9334831237792969, 2.593022346496582, 2.066075086593628, 1.9598454236984253, 2.2611234188079834, 2.3362338542938232, 2.4279520511627197, 2.011563301086426, 1.846527338027954, 2.1489341259002686, 2.2113442420959473, 2.3078370094299316, 2.1257548332214355, 2.445404529571533, 2.261113405227661, 2.363154888153076, 2.398702621459961, 2.0516843795776367, 2.649569034576416], dtype='float32').reshape([24]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53489f1412cbd8573cfd7ebf5c5ac946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(-0.2633531093597412, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a6b9cbf793e741a793335b61e723bbb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([17571], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a6b9cbf793e741a793335b61e723bbb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([17571], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4bb590c1dc345bb54fe3fedd3a46bb93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(1439.754638671875, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_926bc6237f6cdd8946f3837a7b880e9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e159018cb953531937774a04000871d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6601bc42a9c3f7be2745af736c17c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f572c295acf92b2f755359bb0aec223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([16.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58f208baa1620b964905af37f559414e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eaa5b24af245e0260d218c215509f9e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_fc282ce2c0868b98850e2a9bf20e6071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a291cf0e4101be8ef856bb1c11771f1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7df72b4a15f27d8b583ccb3731d38114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0637b058fed81220598bef452adc2047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5af559f72f898cf817b46a32ba7e1eae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4596256ba9599e85f45fe493cc8139d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.3889973759651184]], [[-0.05222040414810181]], [[0.08656024932861328]], [[0.31698358058929443]], [[0.09362101554870605]], [[0.28389525413513184]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3a9a36360edc3ce4b44bad032cabf755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.21859318017959595]], [[-0.11547282338142395]], [[0.30508702993392944]], [[-0.403150737285614]], [[-0.3219117522239685]], [[0.22421681880950928]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2d4cb7c72468d84a7cba2539edaee231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.1871953010559082]], [[0.0767585039138794]], [[-0.4223214387893677]], [[0.46750426292419434]], [[-0.3059464693069458]], [[-0.21983292698860168]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9a111f96cfd84bc4d45276a1f7b6121a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.3016904592514038]], [[0.04326122999191284]], [[-0.16104766726493835]], [[-0.023694336414337158]], [[0.263411283493042]], [[-0.1024852991104126]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4f8b5e81529bad4994fc1877fbf94a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.07609263062477112], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_62c4875a58a74bc660e3c97276905d9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27330920100212097], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0c337aaa7af470487b288a84df16ee04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.08249622583389282], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2d98b99bd7976bd7b326482dd102baf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05669267475605011], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7acd13a6a171ecf11de7cd5843ec4ed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3798350691795349], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3e930baa408db732c30aa0559e56f5dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.25555121898651123], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9dabe9bc666449033524a93dd644818b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.001259535551071167], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_17769a41c6e955db544438f9ab85e8e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2672431468963623], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a4abab9eb9dba9924cc65b9486e4683a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.43146899342536926], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0cc9774ab3c67ebf8a9594de326bf668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06460601091384888], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a4376c0596cc610ce501d2b3fc2e7215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4421810507774353], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d05791587907856551d335c1bdfeb211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3207743167877197], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_70654d3444b9bbd15520cc62aaa177ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3128872513771057], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_91cc4e01dfd7433d7d6e99d212d13697(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.9325518012046814], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0ae21bfab391c3b1d673b7e80a4c818b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.17677366733551025], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5ecbf907c777d02a91ed5b2da16ec430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.447664737701416], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_498e9bf05cc23afd477e9530d4bde279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4452681541442871], dtype='float32').reshape([1]),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3a87c32ddd3eac6f3bf2cdf48dbe1fc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3513146638870239], dtype='float32').reshape([1]),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d335e3229d8cd1b85b912a62c76eca95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32380127906799316], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_24c9ac29fb7a1ce85e17044777aff579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12816721200942993], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcac98a2f4de24b221b63e1c6d1d377f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0850069522857666], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b69e14c7e81850fdc20d5a870a276aff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03722533583641052], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_de2d9fac6ea81526261956beab4e3407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b904837812aa175a0ef11a89cba549ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adc5dc1a9beaedf06881268b87297646
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c7b9f48c8c2811817711b2237a48bbab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_de2d9fac6ea81526261956beab4e3407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a208d54908027a5894d2dbb61aaedb8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c2af6f9fae9ad9af58d6f31ced9ab1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53d62bb0bbe762f7bc89e0e293a7df25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7b6b387f69c2ef985f7832ebe895e6db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_658a37a1d863cf5963ac813f4e938b0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f4699da7c0237c8449b4859ecbd10a
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ae15d9e4db29685fe17a0008415a7d08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4998dcbf805288ff1c5ee6405f608cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_41e0c1de811b31b97dc7d8fb9ae6d804(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fdbda7f418940621c9ce242781a66cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_24f1bb6d71f29f992859e5d694231b72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.08003783226013184], dtype='float32').reshape([1]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7d8a37001d8296c72b727c733faa330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1280692219734192], dtype='float32').reshape([1]),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d335a43e2f68c86c01c587ab2bb70197(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.477338045835495], dtype='float32').reshape([1]),
            paddle.to_tensor([0.05000000074505806], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_13991c683a35ac54050a0ca8a3bb8f83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_367be3469dea6c6b1a20c5774f755539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa7ce3c999b7b8ad235737bb90a4c9f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9024006fea95fe45cf4601048a5d6f89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b10b7e2a7acc3b8ba9e78b0bcf8336c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1027], dtype='int32'),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_76a3523d15ba7a37cc9e3f283d068a56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1027], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_79e4e7a75e34e1a936e47c34f3a6ecb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f04cc37a5a79e4dd940da4f60859c0d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_63878e0c869e6c40e321708feaec42ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_79c152265bc7027ec31ac37ec29d9285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_79c152265bc7027ec31ac37ec29d9285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_75e5272c296e5a04f7a1b9b851c2c248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_06593433b018fabcccbae7feda1d118b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([4204, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a5cd55d3ca9902abbee6b15bf60f7b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4204, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_52446edd8d227e5f2bb1e6d104405861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([4204, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_be4f27c7419456a3fe41e1222f91b303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4204, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_be4f27c7419456a3fe41e1222f91b303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4204, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_651d81b81fd38bd8725496a374508cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([86, 1000], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9a6da9e0f0cc1c5ee706481c879e839f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8848e1ad58943886b8272169035d62bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_da367b03d87589b45768d25121e800ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d4b9a4258c9a83ca8b2a5f00020a9bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3409433f6eac75273c6de7e58cb94cd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.111109972000122], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_18222abeeb632e54e9a5caa7188408c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8f425eb6b53843a27ab08b523ce14c65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4c187533edef487fc20c6422dea92feb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f4699da7c0237c8449b4859ecbd10a
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d504efab29947245201d7a36161b2615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_716e10fb28b676988ec81f040a305d6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0e09753150c5a273d10f7e224c37c1ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3bcc2427ca8198e58284ce269763320b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_91c618c3570cdedefb4a2c851f1da033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_2f1e1d1cb385a386debee3e65c2d3661(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa154ca0b4b19b8cae28382567b2746b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0e09753150c5a273d10f7e224c37c1ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_edc7b134328de552020173c3d567c603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_91c618c3570cdedefb4a2c851f1da033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e772a48121f70aa8dfbe5ade81426a0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(71.69361114501953, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fb947dcc91a380f2443ace6a57d57f9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc85c99ba27e8f632f9351aa12d2f60e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d8979f1b48e7898a2a2728d8f8ecf796(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f4d01cbde66a3d30a9339351124a2c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_392691bfa5fcbe85b28abdfad681760a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e6601bc42a9c3f7be2745af736c17c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fc85c99ba27e8f632f9351aa12d2f60e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_58f208baa1620b964905af37f559414e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e817bfd92546c062f99d6e3849de96e2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f4d01cbde66a3d30a9339351124a2c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_6538848c9218756772da6597a4d5b4cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_26944f42f183cdd31461ae3eeb48fce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_89b2e7b5c2c49cde48f2f6209c43ed76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_89b2e7b5c2c49cde48f2f6209c43ed76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_633b78e1be28fd8a193f184a4e1dfb21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0e5cd706982f87f5449b3fe0d9154483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([4680, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f4d7385466c950b4544f18470c193071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4680, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5f83f3b7b9123cd256a0dd8c49a09f78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([4680, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_524484b04999011abcd9c86a1a59256f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4680, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_524484b04999011abcd9c86a1a59256f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4680, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1eb7e7dcfe33086903bb12a0de6e724e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c92bb2d24d837bfcce5795ca2429954b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([54, 1000], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_27ea7f765e775e5583bbc4fbd5570558(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_27ea7f765e775e5583bbc4fbd5570558(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7411c1ffa08b18b6a6e267707e4ef2f5
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_429498da3cdf63c5069b250b68a01844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_31c0eb7a8fff77ea55c2da76cbaa00b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d9d0b00f6f5adbe073a44081ed81ebf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec48de7d8073d09ef81446122d720d9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3778, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_baa4fcd6a932b5be62066fac029044be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c0873e3bc4dc637a4a7e4fc2210a86
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ebe03ee1ef29006fdc8a9b9d98b5eb3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3778, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ebe03ee1ef29006fdc8a9b9d98b5eb3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42cbe7d21769457608268950030ac570
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3778, 4], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f60d78e4d19e5f19fd199ac3df046071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.17677700519561768], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_d3546200ff7a7cf24002e192267b1ea3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.125], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ecb74b1a0db33dec5f1034260d90fa68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9bb99f5b7d6e5ec5ac77a934b3e3af1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f3f91040ffe46a0c51e297bdb488c9aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_feb571b688d4b2ec86db7cd3626b95f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1d7d3b8bb069f9b68f9ca177c3dc27e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f06a455ad86d70f139c0292d7498a232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_467d226b9847fc100e7bbbfc06fd6780(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_320f2ea1a40371cbcf5743e54c0a1d04
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_ea780090512f9128f779891eb8d8696f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3aa9a1906cbb9d7960b6d6014d185e9d
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_db4b39b63f0420e25dae06be27138f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0af9c566d7705630c992f3fc58768dc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_64902ca8eb35818d7bf0ce33cd97452c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_73f19648454ea59b98601a6a963d0c43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f4699da7c0237c8449b4859ecbd10a
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f59f893beed389c4ee8ae999888c64af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c998d4dce0aee6603b3d084197c94544
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_31f5f21d52f1f429aa0591a46c23a4be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aec9df8215b8632e8426a1cf510bec35
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ffbf372734ba6ca4198f42533c8f73ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c07aa94c472970ba0b30d7d505a7d3fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2.5], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_371fad8f28cee215e380e2eb966eb14d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.41676831245422363]], [[-0.29034411907196045]], [[-0.360625684261322]], [[-0.17076244950294495]], [[-0.10002967715263367]], [[-0.46427422761917114]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_114bac317fb85772f9ab44e79828912f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.42071056365966797]], [[-0.3155045509338379]], [[-0.24923381209373474]], [[0.24133849143981934]], [[0.10318785905838013]], [[-0.05146479606628418]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53f84fac25a621b5f9ec89a81fa1f72d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.3246951699256897]], [[-0.292475163936615]], [[0.20421487092971802]], [[0.21846681833267212]], [[-0.2251012921333313]], [[0.03535884618759155]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a5bad7c43128e0e63dd453ee2678d345(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da9532be7cd6653d265ebcdc1e80a6dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.3679172992706299]], [[-0.4908938407897949]], [[0.18880033493041992]], [[0.4726373553276062]], [[-0.2904278039932251]], [[0.09679967164993286]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([0.20000000298023224], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_197ee2a7316fe22fdd3ef4dbb584d166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_079b021238f4ac5824a617c7f23bc5cd
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[36], dtype='int64'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_95d208be3b26ddfde89c15505e75494c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1308422a2106ec906b9c5c62fc9aecef
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d15a1151264fc90fa66daea5281418e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab374bfab956afa0d1b2f54b51ed69ac
    def get_inputs(self):
        return [
            paddle.to_tensor(0.16510915756225586, dtype='float32').reshape([]),
            paddle.to_tensor([0.25], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c35389f2a5801d975514e8a86968e912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2002], dtype='int32'),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9edf9195a9d8f75cf6701ae7ee1a1a6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2002], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a1dbcb0f82733770ea16e5799ac63021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1021], dtype='int32'),
            paddle.to_tensor([9.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_61eb4118e81dcd422330247642d3baf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa0168e05c33025c4bf0c130979d1ca
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1021], dtype='int32'),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b05f1cee4f59644f08844fd46eeea90a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0aa0931ee3c843e1eeb9868553fd433
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1.0], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()