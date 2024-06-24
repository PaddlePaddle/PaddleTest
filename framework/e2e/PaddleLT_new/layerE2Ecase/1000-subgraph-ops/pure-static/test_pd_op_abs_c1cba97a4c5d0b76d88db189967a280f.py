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



class PrimitiveOp_2e1a7c83c21c85aa6cf7cd533ff36614(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5fd3949bac5582a9af6a29c9817dccac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e1a7c83c21c85aa6cf7cd533ff36614
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5e42c7490339aee257e47ac0070cf768(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e11d10f51a2521eacd9d58e5a751e9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e42c7490339aee257e47ac0070cf768
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e22c707f3957723b1227290d9a80d64a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9a611d2b7d37e7c964dac6cc6d9760b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e22c707f3957723b1227290d9a80d64a
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5da4620298afe229315d5b1aabb58d5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[53, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0cde7f604816a5c8165f76c21680c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5da4620298afe229315d5b1aabb58d5c
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_41a3d4ed25f9fc8294eaf7efd30cc9ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1758, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3206769195be9d4451421c251fb2548(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41a3d4ed25f9fc8294eaf7efd30cc9ba
    def get_inputs(self):
        return [
            paddle.uniform([1758, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6d581d4a7b1a4ddc946563bdcf909282(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4713ff3c104ca0cdf290e700e4dc1b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d581d4a7b1a4ddc946563bdcf909282
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.25677168369293213, -0.06483365595340729, 0.1840917319059372, 0.21405082941055298], [0.131243497133255, 0.286072313785553, 0.14388522505760193, 0.09586738049983978], [-0.44949236512184143, 0.0903753936290741, 0.08646504580974579, 0.10369832813739777], [0.1921076476573944, -0.2811718285083771, 0.062019556760787964, 0.44114696979522705], [-0.005936041474342346, 0.09648439288139343, 0.23188406229019165, -0.2345970869064331]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_fa6c57f2eb7c22be7d7a628b8347a88a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d581d4a7b1a4ddc946563bdcf909282
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1816149801015854, -0.09483715891838074, 0.07430045306682587, 0.05605560541152954], [-0.10881158709526062, -0.026398949325084686, 0.24707551300525665, -0.000649183988571167], [-0.334237277507782, 0.32552292943000793, 0.013933449983596802, -0.08204120397567749], [-0.10881158709526062, -0.026398949325084686, 0.24707551300525665, -0.000649183988571167], [-0.334237277507782, 0.32552292943000793, 0.013933449983596802, -0.08204120397567749]], dtype='float32').reshape([5, 4]),
        ]


class PrimitiveOp_5a61b537fdd5da6e3871c0805e59ae14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b49a395deb1d6f90a82e79dc58cc64aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a61b537fdd5da6e3871c0805e59ae14
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_36f4f4a1a84398c025fc2a30444431b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5593, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4959159e53ff586565ec27ed58b0370a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36f4f4a1a84398c025fc2a30444431b7
    def get_inputs(self):
        return [
            paddle.uniform([5593, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2178e9c120dbd80e3cc79426b0ee0010(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b580072df5d8ffa134dcd02e96810c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2178e9c120dbd80e3cc79426b0ee0010
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05213640630245209, 0.09683552384376526, -0.3227783441543579, 0.0005768388509750366], [0.016525864601135254, 0.354070782661438, -0.051243215799331665, -0.14045239984989166], [-0.14370733499526978, -0.3452792465686798, -0.244966059923172, 0.12276658415794373], [0.016525864601135254, 0.354070782661438, -0.051243215799331665, -0.14045239984989166], [-0.14370733499526978, -0.3452792465686798, -0.244966059923172, 0.12276658415794373], [-0.28111082315444946, -0.2227138876914978, -0.09374728798866272, -0.29603323340415955], [-0.28111082315444946, -0.2227138876914978, -0.09374728798866272, -0.29603323340415955]], dtype='float32').reshape([7, 4]),
        ]


class PrimitiveOp_b86938a5e248737c86607aed172a9aaa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ea1fb9efd1aad525542b8e571e5aca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b86938a5e248737c86607aed172a9aaa
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0dcb72f58c1bf0f6727269af285d7d73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[103, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8a9fc6c2248274900157a0e781589ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dcb72f58c1bf0f6727269af285d7d73
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd63c7231b70123ce5281c0a00502f23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1763, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2f11858a63dc8dbc4fa27d8709e0e0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd63c7231b70123ce5281c0a00502f23
    def get_inputs(self):
        return [
            paddle.uniform([1763, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_70dd88e24e4d7418aa2a7d24b2328d10(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_380b2bff971f1afdd805cc3d5850a7e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70dd88e24e4d7418aa2a7d24b2328d10
    def get_inputs(self):
        return [
            paddle.uniform([1490, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b62f9555ad9c7c4f66aa0f0f18d80500(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93dbecbd4139ff18c611376d05aedec0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b62f9555ad9c7c4f66aa0f0f18d80500
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4721890091896057, 0.13927704095840454, -0.32948899269104004, 0.10651195794343948], [-0.0006179213523864746, 0.0033878684043884277, 0.0574188232421875, -0.054265450686216354], [0.3772001266479492, 0.07573233544826508, 0.05967779457569122, 0.010373055934906006], [0.22724922001361847, -0.15835122764110565, 0.28487300872802734, -0.1891089379787445], [0.22724922001361847, -0.15835122764110565, 0.28487300872802734, -0.1891089379787445], [0.3772001266479492, 0.07573233544826508, 0.05967779457569122, 0.010373055934906006]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_024c1794395c3f36f86d0c9f4472f617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d581d4a7b1a4ddc946563bdcf909282
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0917310118675232, 0.15178906917572021, -0.03986068069934845, -0.002778302878141403], [-0.25007253885269165, -0.07839547097682953, 0.16669723391532898, -0.34016889333724976], [-0.037536993622779846, -0.22154566645622253, -0.2786290645599365, -0.3307667374610901], [0.1383073329925537, -0.31077325344085693, 0.33028000593185425, -0.13818231225013733], [-0.0917310118675232, 0.15178906917572021, -0.03986068069934845, -0.002778302878141403]], dtype='float32').reshape([5, 4]),
        ]


class PrimitiveOp_81f58485e138de1768076195c411358a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ce4553bb26ec44995308a0eb161b28b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81f58485e138de1768076195c411358a
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_614f5b6dd691404e7c87db643cbe5a80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e0b2b74426ef9145e302ffd90e00621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614f5b6dd691404e7c87db643cbe5a80
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2753981351852417, 0.27793630957603455, 0.17279444634914398, -0.1332867443561554], [-0.09361551702022552, -0.1447596698999405, 0.08765412867069244, -0.32762980461120605], [-0.17097410559654236, -0.2685585618019104, 0.0018512457609176636, -0.31080305576324463], [0.12934653460979462, -0.0022540315985679626, 0.2781771421432495, -0.30496832728385925]], dtype='float32').reshape([4, 4]),
        ]


class PrimitiveOp_5a7dc964ab4827838ef16c98d448eeb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[84, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c62c9e28cea62a9df97aee3f1c5538e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a7dc964ab4827838ef16c98d448eeb2
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_da328e46b1bdae9e13a9a1c726c900e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2076, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a062e9b7f06a38a2941ff648cab3ab00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da328e46b1bdae9e13a9a1c726c900e3
    def get_inputs(self):
        return [
            paddle.uniform([2076, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e307f6e251cbb9d3a5a7b86b44cdbae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2178e9c120dbd80e3cc79426b0ee0010
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03508669137954712, -0.16850346326828003, 0.18960177898406982, 0.007154777646064758], [0.03508669137954712, -0.16850346326828003, 0.18960177898406982, 0.007154777646064758], [0.10633885860443115, 0.07276816666126251, 0.029908359050750732, -0.12192053347826004], [0.06507094949483871, -0.243668332695961, 0.21072156727313995, 0.08438259363174438], [-0.11407667398452759, 0.0010586678981781006, 0.3051604628562927, 0.2620214521884918], [0.06873470544815063, 0.0030619800090789795, 0.05197077989578247, -0.09565383195877075], [-0.26225602626800537, 0.06410759687423706, -0.13850826025009155, 0.04361702501773834]], dtype='float32').reshape([7, 4]),
        ]


class PrimitiveOp_7dfb0a11cf57c66c218b7433152ec835(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a61b5afe59bce0b420a1e787178c04b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7dfb0a11cf57c66c218b7433152ec835
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_99d93a50df9d281b22231092485fa0da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d0ebce7c2b779acf7b1e05e03692770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99d93a50df9d281b22231092485fa0da
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dd558cab0f7731f3f9686583680f40fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4642, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_311de2aa36b410378821b1be451d41c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd558cab0f7731f3f9686583680f40fb
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ce8e1c445fe74356e74f14ff21ceb86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1047, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_079b3a2eb6ca0c0b3c4450bc3dde114d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ce8e1c445fe74356e74f14ff21ceb86
    def get_inputs(self):
        return [
            paddle.uniform([1047, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a9630e8545beaddd926653ee516a3956(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b62f9555ad9c7c4f66aa0f0f18d80500
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19283601641654968, 0.44643473625183105, -0.1266147792339325, 0.3170720636844635], [-0.054931044578552246, 0.28975534439086914, -0.2113265097141266, -0.08645415306091309], [-0.054931044578552246, 0.28975534439086914, -0.2113265097141266, -0.08645415306091309], [-0.3403587341308594, 0.34647834300994873, -0.08163797855377197, -0.00804758071899414], [0.24778346717357635, -0.10596618056297302, -0.31448638439178467, 0.1395723819732666], [-0.38121309876441956, 0.2610614597797394, -0.1390228420495987, -0.18141451478004456]], dtype='float32').reshape([6, 4]),
        ]


class PrimitiveOp_380c72d3b13276b5deabe54ea6141818(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 2, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21173c1035dc4ec798c31fc0cf817664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_380c72d3b13276b5deabe54ea6141818
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cb58e13e7c749ce50045faa688f10d8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 2, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53d58881f6978dc7625d999c5a18d7e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb58e13e7c749ce50045faa688f10d8d
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_694545f2b7efe73e654c98068614e16e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04cb9680ba5179b5e370e8d2e0d06c37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_694545f2b7efe73e654c98068614e16e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_90b2cf58938489845e98b9f8fb88612c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2359, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a42c99845ed0d9eaf00be47e621099f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90b2cf58938489845e98b9f8fb88612c
    def get_inputs(self):
        return [
            paddle.uniform([2359, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_920015cbb99b1f396bdeace3bf22984b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3049, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e8fb87273bf32a5fbb72f79fdc1373b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_920015cbb99b1f396bdeace3bf22984b
    def get_inputs(self):
        return [
            paddle.uniform([3049, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_38c20135800d8b7d7b8d69f350e16aee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3806, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a98aa7830983933e22132e82b00588f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38c20135800d8b7d7b8d69f350e16aee
    def get_inputs(self):
        return [
            paddle.uniform([3806, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8155fec171bea0b2c7f2d039e4b9a472(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22cebc46455a5e524b331c2c0a7657fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8155fec171bea0b2c7f2d039e4b9a472
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6dfe8a64cb50bfe2bcb3b76550a033fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be50bde3593e384fb42029edeb9df238(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfe8a64cb50bfe2bcb3b76550a033fe
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_160239381517b9853a36bd639404d4c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[47, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_301241a5cfd71fd36ca4997b8cf2deb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160239381517b9853a36bd639404d4c4
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5664ac8e022b456113c2ffa1d50ad633(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2054, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c2bd59d8294d6a84c3edce6afc9a3cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5664ac8e022b456113c2ffa1d50ad633
    def get_inputs(self):
        return [
            paddle.uniform([2054, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f661a5141cec2d4495e1f81d65e5b436(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_597422cef17b744c96870e47421af816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f661a5141cec2d4495e1f81d65e5b436
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cef9e6f2e0429a84f81b7b693f87f7a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d581d4a7b1a4ddc946563bdcf909282
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00888514518737793, -0.04896102473139763, 0.0483778715133667, -0.03471830487251282], [-0.009788215160369873, 0.05413275957107544, -0.30535221099853516, 0.07475248724222183], [0.39217886328697205, 0.2077333927154541, -0.19593361020088196, 0.054433196783065796], [0.39217886328697205, 0.2077333927154541, -0.19593361020088196, 0.054433196783065796], [0.09205174446105957, -0.13316354155540466, -0.10884398221969604, -0.3186963200569153]], dtype='float32').reshape([5, 4]),
        ]


class PrimitiveOp_290f3d8b3a3e03a240af73601481ebe9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36476e98052e52c5b0c851343fdbdebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_290f3d8b3a3e03a240af73601481ebe9
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f4a3bc6e50a85990e100bb7de43cfb92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4218, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa76a01dff3c1bede1c0b73c4d67961c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a3bc6e50a85990e100bb7de43cfb92
    def get_inputs(self):
        return [
            paddle.uniform([4218, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aab17e4c43effa8149de0fe4dac7c08d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2178e9c120dbd80e3cc79426b0ee0010
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.01474866271018982, 0.24872198700904846, 0.05770754814147949, -0.18724988400936127], [0.09170161187648773, -0.045166343450546265, 0.006407499313354492, 0.06573697924613953], [0.20618627965450287, 0.32045453786849976, -0.020322144031524658, -0.06753784418106079], [0.01474866271018982, 0.24872198700904846, 0.05770754814147949, -0.18724988400936127], [0.2491391897201538, -0.22439298033714294, -0.35869306325912476, -0.05700208991765976], [0.017012417316436768, -0.05802673101425171, 0.03807038068771362, 0.10305461287498474], [0.2491391897201538, -0.22439298033714294, -0.35869306325912476, -0.05700208991765976]], dtype='float32').reshape([7, 4]),
        ]


class PrimitiveOp_b41f0c7a4755f75643a6fbf31fcb7605(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[52, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f65d10b9d6cece8896dabaab7a0e49d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b41f0c7a4755f75643a6fbf31fcb7605
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()