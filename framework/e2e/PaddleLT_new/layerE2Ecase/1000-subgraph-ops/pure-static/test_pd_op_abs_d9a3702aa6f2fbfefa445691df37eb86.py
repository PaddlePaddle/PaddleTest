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


class PrimitiveOp_aa1f3c488a34d5b872c7ea6a2e2dd95b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1786, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37dca61c8f12e5ec66b5bc31583917fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa1f3c488a34d5b872c7ea6a2e2dd95b
    def get_inputs(self):
        return [
            paddle.uniform([1786, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_2c52bd70a4e7fea66f7f4694c610ff88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d581d4a7b1a4ddc946563bdcf909282
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23033301532268524, -0.14335858821868896, -0.2451336830854416, 0.01158994436264038], [0.1772065907716751, 0.3110963702201843, -0.2927146852016449, 0.011528611183166504], [0.24803833663463593, 0.11553388833999634, -0.05100834369659424, -0.2954212725162506], [0.02581791579723358, 0.06242261826992035, 0.31831613183021545, 0.10948888957500458], [-0.1231008768081665, -0.16309943795204163, -0.2740582227706909, -0.014111340045928955]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_25bec4fe7ab9014823d6f79d987f636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d581d4a7b1a4ddc946563bdcf909282
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.15211257338523865, -0.13868305087089539, 0.07743817567825317, -0.17150861024856567], [-0.10281427204608917, -0.09220492839813232, 0.14854612946510315, 0.10860636830329895], [0.09037956595420837, -0.021578580141067505, -0.001748381881043315, -0.06808262318372726], [-0.10281427204608917, -0.09220492839813232, 0.14854612946510315, 0.10860636830329895], [0.09037956595420837, -0.021578580141067505, -0.001748381881043315, -0.06808262318372726]], dtype='float32').reshape([5, 4]),
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


class PrimitiveOp_f45088cc26a10d6f340f17afc6b2400d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5529, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_098449daa278f4f130019d82e9aa11c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f45088cc26a10d6f340f17afc6b2400d
    def get_inputs(self):
        return [
            paddle.uniform([5529, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_5e7c95d76ebc47d8bda44e3bd24af7e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2178e9c120dbd80e3cc79426b0ee0010
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.18005210161209106, -0.014196991920471191, 0.15634046494960785, 0.10224347561597824], [-0.2896423935890198, 0.018510550260543823, 0.03333774209022522, 0.10551036894321442], [0.28359436988830566, -0.09202780574560165, 0.27413296699523926, -0.07617615163326263], [-0.2896423935890198, 0.018510550260543823, 0.03333774209022522, 0.10551036894321442], [0.28359436988830566, -0.09202780574560165, 0.27413296699523926, -0.07617615163326263], [-0.19862128794193268, 0.22186554968357086, 0.21049772202968597, 0.0943182110786438], [-0.19862128794193268, 0.22186554968357086, 0.21049772202968597, 0.0943182110786438]], dtype='float32').reshape([7, 4]),
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


class PrimitiveOp_0a0a283bd9cbbdb4ce7a77de12a2dd8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1767, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e63524e6018d24d5b3e34ab3638ca0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a0a283bd9cbbdb4ce7a77de12a2dd8a
    def get_inputs(self):
        return [
            paddle.uniform([1767, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c09b4981c77c7b37c3a4d5872d5a1b35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b62f9555ad9c7c4f66aa0f0f18d80500
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11206990480422974, -0.19980105757713318, -0.28472182154655457, -0.09120070934295654], [-0.05969715118408203, 0.1464267075061798, 0.09999644756317139, -0.21400588750839233], [0.17537441849708557, -0.40648606419563293, 0.07399022579193115, 0.07907019555568695], [-0.13181909918785095, 0.14245106279850006, 0.021578490734100342, 0.1497858166694641], [-0.13181909918785095, 0.14245106279850006, 0.021578490734100342, 0.1497858166694641], [0.17537441849708557, -0.40648606419563293, 0.07399022579193115, 0.07907019555568695]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_61b03f71435ce879ff0bf0dad01e687f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d581d4a7b1a4ddc946563bdcf909282
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.014901027083396912, 0.004994511604309082, -0.00011786818504333496, -0.18285894393920898], [-0.12429235130548477, 0.0870169848203659, -0.029167458415031433, -0.01861666887998581], [0.11092722415924072, -0.10363040119409561, -0.12077483534812927, -0.10269473493099213], [-0.430207759141922, -0.124886155128479, -0.06110185384750366, 0.010581247508525848], [0.014901027083396912, 0.004994511604309082, -0.00011786818504333496, -0.18285894393920898]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_055ed536e3ec181ee1c12ba78f7d4533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614f5b6dd691404e7c87db643cbe5a80
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3960532248020172, 0.3196747303009033, -0.031348731368780136, 0.14671191573143005], [0.43643319606781006, 0.027652651071548462, 0.11587986350059509, 0.09504431486129761], [-0.05985875427722931, 0.3477330803871155, -0.21412833034992218, 0.27874112129211426], [0.266995906829834, 0.09098441898822784, 0.010822445154190063, 0.062081217765808105]], dtype='float32').reshape([4, 4]),
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


class PrimitiveOp_b4ba97f8e1aacac6daca76be8fcb8682(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2010, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a77998f99e0790642af5cc306c5ee92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ba97f8e1aacac6daca76be8fcb8682
    def get_inputs(self):
        return [
            paddle.uniform([2010, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba272fe382b23c3532e6bfd1f8a11941(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2178e9c120dbd80e3cc79426b0ee0010
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.07020962238311768, 0.22973822057247162, -0.2077721655368805, -0.3234386444091797], [-0.07020962238311768, 0.22973822057247162, -0.2077721655368805, -0.3234386444091797], [-0.1312311887741089, -0.2076951563358307, -0.3393630385398865, 0.20015442371368408], [-0.27245303988456726, 0.004318207502365112, 0.0672927275300026, 0.39573079347610474], [0.14881111681461334, -0.21468481421470642, 0.0474470853805542, -0.06096624210476875], [-0.029213299974799156, -0.4640655219554901, -0.25611305236816406, -0.2912372946739197], [0.006981849670410156, 0.0655575841665268, 0.08795641362667084, -0.09838144481182098]], dtype='float32').reshape([7, 4]),
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


class PrimitiveOp_46413c4520f000ae2681502cf3b1573f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4663, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_058a017d106e2d453f80eae320cb500d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46413c4520f000ae2681502cf3b1573f
    def get_inputs(self):
        return [
            paddle.uniform([4663, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e75b831f6e5caf98bbe5a3804cf5a624(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1090, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6297fe46867c5cf0557f4e5cc699394e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e75b831f6e5caf98bbe5a3804cf5a624
    def get_inputs(self):
        return [
            paddle.uniform([1090, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af48b88c12af31af08967b801d124b08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b62f9555ad9c7c4f66aa0f0f18d80500
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1750708818435669, 0.22437986731529236, -0.24328318238258362, 0.28154340386390686], [-0.04457196593284607, -0.1831628531217575, -0.09731841087341309, 0.33579692244529724], [-0.04457196593284607, -0.1831628531217575, -0.09731841087341309, 0.33579692244529724], [-0.11138305813074112, 0.33443817496299744, 3.0517578125e-05, 0.1521768569946289], [0.0802762508392334, -0.2545437216758728, -0.1444905549287796, 0.30462419986724854], [-0.15836916863918304, 0.22807666659355164, 0.14974167943000793, 0.13632053136825562]], dtype='float32').reshape([6, 4]),
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


class PrimitiveOp_e52860ca8b36ad718c683c8395a57585(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2374, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47013729ccf5ab7d50f1e3553a156bd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e52860ca8b36ad718c683c8395a57585
    def get_inputs(self):
        return [
            paddle.uniform([2374, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d37418bc03e04aa675d5ad131c961680(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3058, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eff6d91ed806ddf024f8bde848f2a1f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d37418bc03e04aa675d5ad131c961680
    def get_inputs(self):
        return [
            paddle.uniform([3058, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8ec9b2636fd5d1607ef1df8ab6dea3ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3793, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d77ef7a14587271d9a74f3e3086099b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ec9b2636fd5d1607ef1df8ab6dea3ab
    def get_inputs(self):
        return [
            paddle.uniform([3793, 4], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_44ff12993bfcbab3b41be90c61e3df5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2042, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59df62cd3de13c744223609f6359816b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ff12993bfcbab3b41be90c61e3df5b
    def get_inputs(self):
        return [
            paddle.uniform([2042, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_4272dca6fc1904a1699f2133f822e9d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d581d4a7b1a4ddc946563bdcf909282
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.09222230315208435, 0.2660036087036133, -0.2386646866798401, 0.482435405254364], [-0.2286505401134491, -0.020706892013549805, 0.09183747321367264, -0.08162347972393036], [0.3744547963142395, -0.11938200891017914, 0.17953604459762573, 0.15103879570960999], [0.3744547963142395, -0.11938200891017914, 0.17953604459762573, 0.15103879570960999], [-0.020409435033798218, -0.09995437413454056, -0.31952378153800964, -0.4160843789577484]], dtype='float32').reshape([5, 4]),
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


class PrimitiveOp_86496ac3a44315edb450957ab7eb1336(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a28a48b91d521a22436de9b00660c60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86496ac3a44315edb450957ab7eb1336
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c311bc435ba6f1d136f6027b05e0c272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2178e9c120dbd80e3cc79426b0ee0010
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.25657403469085693, -0.2911469638347626, 0.2680356800556183, -0.17256245017051697], [-0.27216804027557373, 0.21796457469463348, 0.2322089970111847, -0.1816495954990387], [0.06819947063922882, -0.016032874584197998, -0.10692843794822693, 0.08240850269794464], [0.25657403469085693, -0.2911469638347626, 0.2680356800556183, -0.17256245017051697], [-0.18578732013702393, -0.23836110532283783, 0.03987012803554535, 0.15123692154884338], [0.3416287899017334, 0.1117306500673294, 0.20233027637004852, -0.09810971468687057], [-0.18578732013702393, -0.23836110532283783, 0.03987012803554535, 0.15123692154884338]], dtype='float32').reshape([7, 4]),
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