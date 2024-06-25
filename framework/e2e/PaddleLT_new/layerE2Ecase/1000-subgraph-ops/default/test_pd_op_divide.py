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



class PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07bcf08a2d2fcb66983dfec5780923d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d6ebd010a5bb57e1b0bc1cf9a08133b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6b40b81719b4ae051a3ad01c76c48ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24af4fac2902b6151c51a988125bcc35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b40b81719b4ae051a3ad01c76c48ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2414522022008896]]], dtype='float32').reshape([1, 1, 1]),
        ]


class PrimitiveOp_16d8bd42f7eff78b9e3230066b7a7608(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[12096, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[12096, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54db2d2e7dc04d7320df17202705f995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16d8bd42f7eff78b9e3230066b7a7608
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([12096, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_309f51c50c9850cd16dd3894e8a41433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7847848f8dae9ca754775f32e227052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_703b9c4a7e8742899c426a973f3f855f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c50da536e457384f20896701b7799dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_703b9c4a7e8742899c426a973f3f855f
    def get_inputs(self):
        return [
            paddle.to_tensor([1109.287109375], dtype='float32').reshape([1]),
            paddle.to_tensor(8732.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b98a10f9217760bd5782eca3497ac30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[6.14845987456647e-07], [0.0007193853962235153], [0.0009237104677595198], [0.0006928475922904909], [0.003465956775471568], [6.337551167234778e-05]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_1bf035ffd4193c4a01f4e1a1e04e3846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.0002494629588909447], [1.4614892279496416e-05], [1.9597899154177867e-05], [0.0011073161149397492], [0.0013705360470339656], [3.511129762046039e-05]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_123f5c92545119c55f3fd5d6f1c04de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b40b81719b4ae051a3ad01c76c48ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.0028592136222869158], [0.023945428431034088], [0.03983227536082268], [0.010959853418171406], [0.1330224573612213], [0.0011201032903045416]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_cb7948835bad5bd55157e3be8dc9c037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f69ac909d4915371a12db1de0ee05cc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07bcf08a2d2fcb66983dfec5780923d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc712d615bb205a08ac859dc5b80a517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c7c07c53a0fe3540238159515a9fda69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4efe19c0d6e5878a9806c7f9f34f96cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(10.010570526123047, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8b495bbecd83c6492b96987d5917d068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(2.6111321449279785, dtype='float32').reshape([]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e4e8ccbae22d55caec42f8915a479c02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c7fcc6f67d43fdb0660a50eae295f92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c7fcc6f67d43fdb0660a50eae295f92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dfd27553c2583a70df1099fcf415792a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-229751.453125, dtype='float32').reshape([]),
            paddle.to_tensor([0.47629663348197937], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a37dc3113bd186f45362d22a26838e9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(99828.9375, dtype='float32').reshape([]),
            paddle.to_tensor([0.47629663348197937], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6ba3e59bbc7dadaaf0597574788153c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(958.2197875976562, dtype='float32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9871bf1396c8b9db33c6571fcb21ce8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364055f7814905dc4901270efaeb86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_345d2d085813649aee260a0dfb3ff397(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5376, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5376, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab9ff37f33666ce96947a55c3c9dee12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_345d2d085813649aee260a0dfb3ff397
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([5376, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c80361c71d4987f902260c9caae754f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc39eb9f4f10134caf76738aca0515f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f81acf25a05f7239fdfe8b66a1ac1491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0729246512055397], [0.034768473356962204], [-0.013078576885163784], [0.16130442917346954], [-0.010292893275618553], [0.12493808567523956], [-0.0019259939435869455], [-0.02474241331219673], [-0.0016566825797781348]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_27fc30483c155ac3655278ecac119864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06701730936765671], [-0.0360693633556366], [0.05359303951263428], [-0.08947645872831345], [0.03258062154054642], [-0.12430624663829803], [0.012700259685516357], [0.037587396800518036], [0.06804581731557846]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.005907343700528145], [-0.0013008894165977836], [0.04051446169614792], [0.0718279704451561], [0.022287728264927864], [0.0006318385130725801], [0.010774265974760056], [0.012844983488321304], [0.06638913601636887]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_c3cad1f8cbce2076a869387e82b80ccd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_465a66e2ae90b90f44749fa6f593413c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3cad1f8cbce2076a869387e82b80ccd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4963630139827728], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_893ef6bc39751381f0b1436dfb526449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_893ef6bc39751381f0b1436dfb526449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ed7c322094c2a3cddee397358d3a322(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(11219.791015625, dtype='float32').reshape([]),
            paddle.to_tensor([0.04751402139663696], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d04b0bf526d54614c734683003178fa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(3932.84814453125, dtype='float32').reshape([]),
            paddle.to_tensor([0.04751402139663696], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ff10856afc23f7aff18e2b675cc93008(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e72de28cdec9793f156b03bdb1df5bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.0, -0.0, 0.0, -0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.004256535787135363, -0.10495690256357193, 0.0012650515418499708, 0.014308074489235878, -0.026299387216567993, -0.016757093369960785], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ce5c73263b96a8d0e32fbda4236989a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1436895877122879, 0.0390329584479332, 0.05066336691379547, 0.12593135237693787, 0.0014194230316206813, 0.05388146638870239], dtype='float32').reshape([6]),
            paddle.to_tensor([0.24177946150302887, 0.14634716510772705, 0.04276227205991745, 0.17142243683338165, 0.07123421132564545, 0.1578609198331833], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ae8ffd959cd5a431e49023bb83c1c7cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.03985670953989029, 0.3563137352466583, 0.0256974995136261, -0.07077597081661224, 0.2433735728263855, 0.24213233590126038], dtype='float32').reshape([6]),
            paddle.to_tensor([0.10679596662521362, -0.2945631742477417, 0.04922857880592346, -0.2021600604057312, -0.10806180536746979, -0.09570963680744171], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_8ed44df149e957ab3fa5959ecb25eade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.008161276578903198, -0.0208590030670166, 0.1359562873840332, -0.3296777009963989, -0.1270461231470108, 0.09199276566505432], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.27261829376220703, -0.09261831641197205, -0.1817198544740677, -0.3106933534145355, 0.10956054925918579, 0.06975878775119781], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_2f4b461057da2f85915fec88371d05d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.043407708406448364, 0.4917392134666443, 0.5114994049072266, 0.0927068218588829, 0.03497505187988281, 1.8152750730514526], dtype='float32').reshape([6]),
            paddle.to_tensor([1.043407678604126, 1.491739273071289, 1.5114994049072266, 1.0927067995071411, 1.0349750518798828, 2.815275192260742], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ded7a171f04bd0407bda87186a45c888(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ded7a171f04bd0407bda87186a45c888(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e6d57a8648eb269edeba7fc36425adf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(20863.6875, dtype='float32').reshape([]),
            paddle.to_tensor([0.2124483734369278], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c5b13ab916ddf2e4367b1f65087e155d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(104527.6875, dtype='float32').reshape([]),
            paddle.to_tensor([0.2124483734369278], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_892e542e26b1d3e6d850cf929ea24c7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(937.0189208984375, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4943d5957299860748a46c4b99879007(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[8400, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c4f36bb817eeb396d8999863494d089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4943d5957299860748a46c4b99879007
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_290e661b966320253ee5661b0d1b2447(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a082489cf7170f8e3177e31458374a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_290e661b966320253ee5661b0d1b2447
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7847848f8dae9ca754775f32e227052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364055f7814905dc4901270efaeb86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b29f2dd204ee3ae0f08950a33ed93d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b29f2dd204ee3ae0f08950a33ed93d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df27cf4e2001fba86550a49c502d2afc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(35052.86328125, dtype='float32').reshape([]),
            paddle.to_tensor([0.43615221977233887], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_869b87f6b071612b0e4bd6f2d119b707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(85329.9609375, dtype='float32').reshape([]),
            paddle.to_tensor([0.43615221977233887], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_608d597b0691cc50d344c156c307c78e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b40b81719b4ae051a3ad01c76c48ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24687178432941437], [0.2461656779050827]]], dtype='float32').reshape([1, 2, 1]),
        ]


class TestPrimitiveOp_f69ac909d4915371a12db1de0ee05cc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d6ebd010a5bb57e1b0bc1cf9a08133b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5c2db118248adf7d70d01bd200c4c91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.03686540946364403]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_055e099b255e03926a23580e3eedc481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.021455012261867523]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.05832042172551155]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_b9777f7e84d97e8588d49a82a9786899(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.010928520932793617], [0.022338353097438812], [-0.05425577610731125], [-0.0027498090639710426], [0.046230580657720566], [0.005657250061631203]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_bef3d915ddfb173abb3f9c722c050e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.031192826107144356], [-0.03952675685286522], [0.07905479520559311], [0.06982093304395676], [0.0003261752426624298], [0.0769810676574707]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.02026430517435074], [-0.017188403755426407], [0.02479902096092701], [0.06707112491130829], [0.046556755900382996], [0.08263831585645676]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_04b3ffbead6ae914b58c1bf626aae3ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b40b81719b4ae051a3ad01c76c48ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24469703435897827]]], dtype='float32').reshape([1, 1, 1]),
        ]


class PrimitiveOp_0179b907d94371262b6921abc8e0258e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4781f35684ad9337ae0342ccc76bd7be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0179b907d94371262b6921abc8e0258e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9871bf1396c8b9db33c6571fcb21ce8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_309f51c50c9850cd16dd3894e8a41433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0dde5b123a5afcfe3f460476eed830b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(58.158084869384766, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_95ab93cc8bb35527f4b01bf19540cc21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(556.2449951171875, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_476b0e48414a5bc3a891c206fc57e3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_476b0e48414a5bc3a891c206fc57e3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_22c0f230be59b870eccfc3827e9ce1fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-419093.03125, dtype='float32').reshape([]),
            paddle.to_tensor([0.22077412903308868], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e924ec604c6653a6c01a6b8f78619196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(118232.71875, dtype='float32').reshape([]),
            paddle.to_tensor([0.22077412903308868], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d633f1c4334c73ad9a8f47eb69c9ea1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0179b907d94371262b6921abc8e0258e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9fd93520018158bb64f4062d3c180d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca0b1640351529ce5c317e86652455c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3cad1f8cbce2076a869387e82b80ccd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0507081113755703], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_26a4d50b038060d8d33d37cbaa906e85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26a4d50b038060d8d33d37cbaa906e85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b737489577651c123682150344f0bea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(693887.75, dtype='float32').reshape([]),
            paddle.to_tensor([0.2323600947856903], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8b1e09a5633ad005fd133b90532967bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(260584.109375, dtype='float32').reshape([]),
            paddle.to_tensor([0.2323600947856903], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cb7ed4f2554f777f08cbf2efd675817e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_703b9c4a7e8742899c426a973f3f855f
    def get_inputs(self):
        return [
            paddle.to_tensor([301.00640869140625], dtype='float32').reshape([1]),
            paddle.to_tensor(2434.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_acab2b587f17dce29045f8a317451586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acab2b587f17dce29045f8a317451586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e89d99b8bce26d2a350ae84febc04ffb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-9207.650390625, dtype='float32').reshape([]),
            paddle.to_tensor([0.43028029799461365], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8aaa154f6feb580308c07589ea046405(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(14978.7021484375, dtype='float32').reshape([]),
            paddle.to_tensor([0.43028029799461365], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cb7948835bad5bd55157e3be8dc9c037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bdc8734d1b0e6c8c8ceeb3459558ebc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[100, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76f6b4224d683a52f5f0a5d81209ffb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdc8734d1b0e6c8c8ceeb3459558ebc7
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffd674483e8ad498f752bedcc9feaa35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3091023564338684, 0.0753689780831337, 0.27828872203826904, 0.2435876578092575], [0.03600621968507767, 0.027483966201543808, 0.16106292605400085, 0.4283529818058014]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.4766256809234619, 0.22583802044391632, 0.2942560613155365, 0.1891525387763977], [0.3694056570529938, 0.10931017249822617, 0.3292573392391205, 0.38082703948020935]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_9c80361c71d4987f902260c9caae754f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7a621c5d797a61a537bd3d1833d32e41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[6069, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9938de2e816ed3abe153d044274abb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a621c5d797a61a537bd3d1833d32e41
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1807c3cf3d5842f56f647db6c5ae78bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[300, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fae332da9bbd2c06fb1eba12cba37fb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1807c3cf3d5842f56f647db6c5ae78bc
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ba2cfe01407f74bfe0bf27797fd30e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05090628191828728, 0.2558620274066925, 0.3996395170688629, 0.24939565360546112], [0.2429737150669098, 0.1858176290988922, 0.3840314447879791, 0.091161347925663]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.4586729109287262, 0.4708313047885895, 0.47626203298568726, 0.10705063492059708], [0.14894841611385345, 0.09577310085296631, 0.1100393682718277, 0.2702256739139557]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_b7a32539c8f5c61032fc32679d04a69d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.17713594436645508], [0.01181112602353096], [0.010422749444842339], [0.025402620434761047], [-0.03610027953982353]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_93532d7092db30b666099d9abf70bf0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06330002844333649], [0.031197108328342438], [0.06478263437747955], [0.02158530429005623], [0.0541597381234169]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.11383591592311859], [0.0430082343518734], [0.07520538568496704], [0.046987924724817276], [0.018059460446238518]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_47b2e15a952a91f081f22c683cef4506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3cad1f8cbce2076a869387e82b80ccd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03637641668319702], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e4e8ccbae22d55caec42f8915a479c02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc39eb9f4f10134caf76738aca0515f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_344a733430d0cd3a35616acc0d219c23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e16ef44e7b6f045c0fd01864839df15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e16ef44e7b6f045c0fd01864839df15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_625bb580ec3cc15f82d973af69c40a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-55658.22265625, dtype='float32').reshape([]),
            paddle.to_tensor([0.0651378482580185], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a2f2d7249750b4786a7d651d9a985ed9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(136041.03125, dtype='float32').reshape([]),
            paddle.to_tensor([0.0651378482580185], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0aa8aac6d277caf396b074f70f7fa06e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0aa8aac6d277caf396b074f70f7fa06e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b510a238253e9255f3ae38903679392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(152087.5, dtype='float32').reshape([]),
            paddle.to_tensor([0.03384774550795555], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_32363ec32a397260039d775fdbafdeed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(175322.09375, dtype='float32').reshape([]),
            paddle.to_tensor([0.03384774550795555], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1a31f10381af44468484386b7b43df65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a31f10381af44468484386b7b43df65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e10fb05678989ce4ac35aeae6803e08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(7177361.5, dtype='float32').reshape([]),
            paddle.to_tensor([0.3044215142726898], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_70fa10c57ae718356dfd892e7142b06e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(213056.90625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3044215142726898], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b7e8a43a454bdf366ac909fed5725967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3cad1f8cbce2076a869387e82b80ccd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.42968687415122986], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9fd93520018158bb64f4062d3c180d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2af09b9341564f3ed3b6b031d641e92c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(14.959243774414062, dtype='float32').reshape([]),
            paddle.to_tensor([3.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_210b6b79c731e4aba93f558e9047993b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20267, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[20267, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d502cef41ffb864084ed36bfb5a2580d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_210b6b79c731e4aba93f558e9047993b
    def get_inputs(self):
        return [
            paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([20267, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d5d5c364c31fa43ba7114f303298752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.08053532242774963], [0.003142738714814186], [0.08117058873176575], [-0.04404306039214134]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_8832f4153ccde1f721fc15c47838794e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.07502203434705734], [-0.009997274726629257], [-0.09352003037929535], [0.06133749336004257]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.005513290874660015], [-0.006854535546153784], [-0.012349440716207027], [0.01729443110525608]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_b644349599b4faa60d8bf3ad3d22af10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(4.391937255859375, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a5cdbdc58ac714eea41c0cc3ef38fb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a5cdbdc58ac714eea41c0cc3ef38fb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10805d0fc4231528cf74be53fefc639e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(171620.421875, dtype='float32').reshape([]),
            paddle.to_tensor([0.006759249605238438], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_73b2728b98dff22d9ccd3eeeffa9ea3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(28925.1171875, dtype='float32').reshape([]),
            paddle.to_tensor([0.006759249605238438], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3e4d7471fa459471b45ad493f187c8c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3cad1f8cbce2076a869387e82b80ccd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09570518881082535], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ce7c0ed6205c5ef1a01ec16d8bb1b41c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(34.254676818847656, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7e1f1271690e70375c15c3d7e621d6eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6804, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[6804, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69b93758a789ef7907850304ca99e29a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e1f1271690e70375c15c3d7e621d6eb
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6804, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_344a733430d0cd3a35616acc0d219c23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44e1fe9cf6eb850794dc4a4bbc88ceac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(242.5929718017578, dtype='float32').reshape([]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b21a5f0bb6d9762f759d96c18a611b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(137.46116638183594, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cc712d615bb205a08ac859dc5b80a517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fb6aed1d3ff583ef33396027c0c1636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8be4838d3b454af4846fe2b6c60b5085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8be4838d3b454af4846fe2b6c60b5085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b62ff7a02aff46a2c1050a85155aa9db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(699384.4375, dtype='float32').reshape([]),
            paddle.to_tensor([0.10235259681940079], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_23023450199f4abcd00645f5c2fad0bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(239146.953125, dtype='float32').reshape([]),
            paddle.to_tensor([0.10235259681940079], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1fb6aed1d3ff583ef33396027c0c1636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()