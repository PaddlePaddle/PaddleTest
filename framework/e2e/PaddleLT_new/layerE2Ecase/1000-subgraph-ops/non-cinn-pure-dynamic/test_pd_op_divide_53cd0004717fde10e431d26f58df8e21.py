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


class TestPrimitiveOp_3f6410354749405d81dd8f149812c2ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24425794184207916]]], dtype='float32').reshape([1, 1, 1]),
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


class TestPrimitiveOp_49e0d10ad1dbacff76977598eb08e0d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
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


class TestPrimitiveOp_905ea5535a06bf93af096d76067c0bce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_703b9c4a7e8742899c426a973f3f855f
    def get_inputs(self):
        return [
            paddle.to_tensor([1073.903564453125], dtype='float32').reshape([1]),
            paddle.to_tensor(8732.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_49c79d3d2e93a45e0fb9621e9f9f2567(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.0010424094507470727], [3.965494761359878e-05], [0.006888340227305889], [0.0026569690089672804], [0.0078109074383974075], [0.0021921955049037933]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_b942252b46d5cc569962960ee755deba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.008773253299295902], [0.014855301938951015], [0.009202162735164165], [0.0006451205117627978], [0.004761108197271824], [0.002814524108543992]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_4738920bae6807c9ccdef438bb1998cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.14872200787067413], [0.1762605756521225], [0.039102714508771896], [0.1306639164686203], [0.11861411482095718], [0.06950155645608902]]], dtype='float32').reshape([1, 6, 1]),
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


class TestPrimitiveOp_1a0a54d876bc7f1b692cf3890033a053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(9.248589515686035, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_70cf2b51f2f423b84db040fa6c5e362f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(2.4917116165161133, dtype='float32').reshape([]),
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


class TestPrimitiveOp_3317575d7d1df3c168fb7e0bc21854b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3317575d7d1df3c168fb7e0bc21854b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ac14beba53fa4e7ad3ee5aa10fde487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(1145334.625, dtype='float32').reshape([]),
            paddle.to_tensor([0.005009666085243225], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e5a7d35eb82c390d02a4360e21375868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(101507.34375, dtype='float32').reshape([]),
            paddle.to_tensor([0.005009666085243225], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b074facb30f09e1a558f76949668a114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(938.126708984375, dtype='float32').reshape([]),
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


class TestPrimitiveOp_eb9d6efea643bf5df74687e22793215f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
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


class TestPrimitiveOp_ee2930fb67de957ea0f28e08acb597ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0027843876741826534], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.06219828128814697], [-0.014282139018177986], [-0.10170114040374756], [0.03550627827644348], [0.019529074430465698], [0.08100476115942001], [-0.07937122881412506], [0.03876090049743652], [0.0033347271382808685]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_c6a29ddc7bb3ae6e7256a661b4320231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03848124295473099], [0.04935537278652191], [0.032189421355724335], [0.03157537430524826], [-0.0032325778156518936], [-0.08090022951364517], [0.10166627168655396], [-0.03094298578798771], [0.0008855792693793774]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.10067952424287796], [0.035073231905698776], [-0.06951171904802322], [0.06708165258169174], [0.016296496614813805], [0.00010452872083988041], [0.022295046597719193], [0.007817914709448814], [0.004220306407660246]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4eda0188ad50ed70412ad078751187d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3377941846847534], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcc65e68f00eadeb6ce1b6db267b8d84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcc65e68f00eadeb6ce1b6db267b8d84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e23d262677e5d35fd3e6b27b03353dc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(12413.376953125, dtype='float32').reshape([]),
            paddle.to_tensor([0.1983392983675003], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_52ff7f69eb8497fa6921e27a91ff849f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(3924.4580078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.1983392983675003], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_fbff9b5cdf986ad9002f6e04e4be8f8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.0, 0.0, 0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.0919826328754425, 0.0004508087004069239, -0.020791757851839066, -0.00025404433836229146, 0.01552492007613182, -0.0659312754869461], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_880655514aec2e9c71e8e362d2269997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.041762497276067734, 0.07953738421201706, 0.004688776098191738, 0.01329649519175291, 0.04933386668562889, 0.05179619789123535], dtype='float32').reshape([6]),
            paddle.to_tensor([0.06237894296646118, 0.11125954240560532, 0.12761320173740387, 0.0912887379527092, 0.08598051220178604, 0.140384703874588], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_12a6f91117f28c89616f9d807925ddf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3682873845100403, -0.004269659519195557, -0.35324352979660034, 0.2066400796175003, -0.21789249777793884, 0.3506285548210144], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2497577667236328, -0.10558420419692993, 0.058859556913375854, -0.001229405403137207, -0.07125036418437958, -0.18803738057613373], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_c1d3560440d5dd436ffa30e3f3a58aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.08779201656579971, 0.13501891493797302, 0.29192203283309937, -0.2597283124923706, -0.2893482744693756, -0.11424572020769119], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.35173314809799194, -0.19269569218158722, -0.4672316908836365, 0.13720351457595825, 0.2932243347167969, -0.05602937936782837], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_7d4a701634a76680947546fbb219d755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6027116179466248, 0.17207784950733185, 0.29091882705688477, 0.09339641034603119, 1.6759088039398193, 1.9497876167297363], dtype='float32').reshape([6]),
            paddle.to_tensor([1.6027116775512695, 1.1720778942108154, 1.2909188270568848, 1.0933964252471924, 2.6759088039398193, 2.9497876167297363], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_54aae011466b840c018c3c1934a3da29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54aae011466b840c018c3c1934a3da29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5553bf5e91d4ec4a57df82d72621c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(327176.53125, dtype='float32').reshape([]),
            paddle.to_tensor([0.2751551866531372], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f72af9769adb9fe305d9e0275e3945ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(101738.140625, dtype='float32').reshape([]),
            paddle.to_tensor([0.2751551866531372], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_13c263a11fa63110fcb2f2b084d77d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(964.1157836914062, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f6d26b909ba0a914711aa1e61eeca30b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_81763756a4de0a896f491ccd483c5ee3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f53ed43e183f6f02e9e10622b37f255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
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


class TestPrimitiveOp_8ef2473d9eaaff60bf2b2ffdc2d2404a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ef2473d9eaaff60bf2b2ffdc2d2404a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ad75928ace23e9314cfa463e0c52e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(155843.03125, dtype='float32').reshape([]),
            paddle.to_tensor([0.3442065119743347], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0628c7e975643e3b16b502c2a2ee34b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(88920.5390625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3442065119743347], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_98a6e4c1cf90fa1bd7b6c90524bb9ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24382051825523376], [0.2482498586177826]]], dtype='float32').reshape([1, 2, 1]),
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


class TestPrimitiveOp_14e74e940ff6276e1836b7404cd3bea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.020566539838910103]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_19cd0728839b2d871e90f6e6670efb2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14905905723571777]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.12849251925945282]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_c69cabdc7c5f9696f01127339573b43d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.01087585836648941], [0.013842154294252396], [-0.04264052212238312], [0.06212622672319412], [-0.0019288918701931834], [0.022789228707551956]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_817f42b1bd4bf4acd4c80bbe5c6a849f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.04338522255420685], [0.04484258219599724], [0.0175580233335495], [-0.05306601524353027], [0.02572479099035263], [-0.03803318738937378]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.03250936418771744], [0.058684736490249634], [-0.025082498788833618], [0.009060210548341274], [0.023795899003744125], [-0.015243959613144398]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_d06f0e3df4150381709c42427123949b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24200740456581116]]], dtype='float32').reshape([1, 1, 1]),
        ]


class TestPrimitiveOp_d3c47f25e9b4fb864c73fcc4b446a37c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
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


class TestPrimitiveOp_5c3401e7b03e80f995c7e41a0649d6c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(62.47104263305664, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_67f4ef63c75b5052a506f282f495d324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(548.83056640625, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b3182e80ec71bf41e8ced1bc489baaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3182e80ec71bf41e8ced1bc489baaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_565a71dbc569d2b15494d51075321141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-124221.953125, dtype='float32').reshape([]),
            paddle.to_tensor([0.4492727518081665], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4e018b5c1cf5793e8c3f145d3075dda3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(115587.796875, dtype='float32').reshape([]),
            paddle.to_tensor([0.4492727518081665], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fd25ef24ed252b29527539cdc530dff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
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


class TestPrimitiveOp_932517480aeeb4ea561ff652d3a660e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03681779280304909], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5e9771f489b8bb88d1b27e6929f52e33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e9771f489b8bb88d1b27e6929f52e33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_076268d673bce656ef54a082eff8ce61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-1692122.875, dtype='float32').reshape([]),
            paddle.to_tensor([0.04309903457760811], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c627bf8fbdae4a5807754ad6a2bdfb81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(265483.875, dtype='float32').reshape([]),
            paddle.to_tensor([0.04309903457760811], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ff7608124c8ff4bf6a6176c10118b781(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_703b9c4a7e8742899c426a973f3f855f
    def get_inputs(self):
        return [
            paddle.to_tensor([310.3299865722656], dtype='float32').reshape([1]),
            paddle.to_tensor(2434.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_46d0a94958a44b2cb186dfcd8052df8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46d0a94958a44b2cb186dfcd8052df8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34f6ef069c798ca2827c98523bf1251e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-41732.37890625, dtype='float32').reshape([]),
            paddle.to_tensor([0.47746357321739197], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fa410bfd9259100a747517da60f7d093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(14965.4912109375, dtype='float32').reshape([]),
            paddle.to_tensor([0.47746357321739197], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cb7948835bad5bd55157e3be8dc9c037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2efa3b98bfee3abd25f1777188941b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39e2b70091966f370195e679487e67e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3836272358894348, 0.15733161568641663, 0.3162004351615906, 0.020395992323756218], [0.25818589329719543, 0.18684004247188568, 0.22731760144233704, 0.46488475799560547]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.4748809039592743, 0.4220361113548279, 0.3332761526107788, 0.11973464488983154], [0.17293894290924072, 0.0714518278837204, 0.24987421929836273, 0.12755435705184937]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_9c80361c71d4987f902260c9caae754f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac9a95d3deba020a96ca1058630f72ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_279e12a9671d6eca5152897f5a55fcbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4e096022da11d53d3c3149c3a3041f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1939072459936142, 0.15718156099319458, 0.43384191393852234, 0.1945497840642929], [0.02189807780086994, 0.11839199811220169, 0.13850662112236023, 0.23024994134902954]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.48236873745918274, 0.3725852966308594, 0.209899440407753, 0.10508774220943451], [0.44159653782844543, 0.4794985055923462, 0.33537226915359497, 0.15888024866580963]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_2bcb25ac38a57b8f606785bfbf88db74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.05804389715194702], [0.03965532407164574], [0.025474827736616135], [0.08548954129219055], [-0.16984909772872925]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_703bf6e26687dbacbdd65224a216c15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07086572051048279], [-0.03364939987659454], [0.027501873672008514], [-0.0839400440454483], [0.10809578746557236]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.012821821495890617], [0.006005924195051193], [0.05297670140862465], [0.0015494965482503176], [-0.06175331026315689]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_fd57b3296a80f6fdcad9812d85d7c567(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10641434043645859], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_6cba6f70564ef31e60619e9c45216785(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cba6f70564ef31e60619e9c45216785(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee472a354702cb588e7141f93c3bb3b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-823989.75, dtype='float32').reshape([]),
            paddle.to_tensor([0.4280586838722229], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cb678ed5b1d758d3511295fdeec70df7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(135582.078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.4280586838722229], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ec9ca067934b8a32250667b051187f38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec9ca067934b8a32250667b051187f38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75d9fa9aa9422a7a6823298061682935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(2096699.625, dtype='float32').reshape([]),
            paddle.to_tensor([0.12333913147449493], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_15d8bad38fb352f27465995fee2e18c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(176722.921875, dtype='float32').reshape([]),
            paddle.to_tensor([0.12333913147449493], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_38ac2c57d39572ef77c9d72731f81383(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38ac2c57d39572ef77c9d72731f81383(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_231f25cda2bf61e262ba09fa1a580ca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(472603.5, dtype='float32').reshape([]),
            paddle.to_tensor([0.011008227244019508], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8d7b399db68decca6cda68ef09fb32fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(218022.875, dtype='float32').reshape([]),
            paddle.to_tensor([0.011008227244019508], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d819ecfe533eb479837d0891793e092f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0030059153214097023], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9fd93520018158bb64f4062d3c180d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd8b41f3fc898082dfd5dcc58d13a9eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(15.697975158691406, dtype='float32').reshape([]),
            paddle.to_tensor([3.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_891fc3a0ea6ba9ba9002c2f1ddd3bb69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([20267, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd8b42a3417cbde66285fc55a66b9fc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.0026173554360866547], [-0.005801578052341938], [0.0918298214673996], [0.14991185069084167]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_d98b73d82bc2f0ddee8708fff46a9a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.017783334478735924], [0.0024686504621058702], [-0.09108061343431473], [-0.10280288755893707]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.015165979042649269], [-0.0033329275902360678], [0.0007492094300687313], [0.0471089668571949]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_5a87c49418b1f35fdd6b9062733d3e08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(4.205384254455566, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7f7cab9b273c10de106a5bc942aac428(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f7cab9b273c10de106a5bc942aac428(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bedaab82e38ae78cba9ae434563018ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-3747.69140625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3375100791454315], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0cfa8592419c51ec45dea7742684fd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(29615.70703125, dtype='float32').reshape([]),
            paddle.to_tensor([0.3375100791454315], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_59094da38591619e936205edf0cbf4ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.42677927017211914], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0aa5e70ef314a737de7e55ef487a6d83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(35.837547302246094, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_db88b631d7f5d0d057e74c91c8b830fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
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


class TestPrimitiveOp_29ba62729e075f1defcfc049bdb40abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(240.34210205078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_21942f22ff5ae2383271f9f3bf7a9a24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(137.7935028076172, dtype='float32').reshape([]),
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


class TestPrimitiveOp_09109f417cf16acf844b8c5a83bbfaa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09109f417cf16acf844b8c5a83bbfaa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_439124b59cce4fa92d40c185beb08024(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(178604.875, dtype='float32').reshape([]),
            paddle.to_tensor([0.31607964634895325], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c83742c4a2afac0dac86afdafc036c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(243006.140625, dtype='float32').reshape([]),
            paddle.to_tensor([0.31607964634895325], dtype='float32').reshape([1]),
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