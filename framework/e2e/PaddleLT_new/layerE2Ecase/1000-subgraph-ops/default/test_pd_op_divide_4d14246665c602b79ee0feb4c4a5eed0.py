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


class TestPrimitiveOp_f2c55d12f9598456fea40d7c07ad8844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b40b81719b4ae051a3ad01c76c48ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24566441774368286]]], dtype='float32').reshape([1, 1, 1]),
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


class TestPrimitiveOp_576aaacadaa199b36fe42c9fc60c5750(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_703b9c4a7e8742899c426a973f3f855f
    def get_inputs(self):
        return [
            paddle.to_tensor([1106.7281494140625], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_4e6d37e5da06e80d86dac1aba13ea255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.001488814945332706], [0.0001351384853478521], [0.055184073746204376], [0.0012555437861010432], [0.01362524088472128], [7.094441116350936e-06]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_19aa07f861bdfc393e20fafd7f6b30cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.002299762098118663], [4.731879380415194e-05], [0.02808908373117447], [0.005427603144198656], [0.0045984890311956406], [7.206248119473457e-05]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_511340362236980dbefc522cf6e92144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b40b81719b4ae051a3ad01c76c48ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.07035422325134277], [0.03182988613843918], [0.2457665055990219], [0.055125631392002106], [0.035247135907411575], [0.0004214059445075691]]], dtype='float32').reshape([1, 6, 1]),
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


class TestPrimitiveOp_15c1f5ace71741e0d533ceb3d1edc468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(9.74549674987793, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ddb320c047df9be5ef0219819d4817e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(2.20776629447937, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a86634eb49a8094d14f35a6f4e2cab00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a86634eb49a8094d14f35a6f4e2cab00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df6aab88ab0d963237760d4acecc527b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(4317712.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.33612334728240967], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b0f760039651f0608f981ff45b630f1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(104618.0625, dtype='float32').reshape([]),
            paddle.to_tensor([0.33612334728240967], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_94310c464e7e9d10ded4e83670feca6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(947.781005859375, dtype='float32').reshape([]),
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


class TestPrimitiveOp_ff2df445197ee55368636cd35587a2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.04838085174560547], [-0.056969888508319855], [0.065152607858181], [-0.0019278853433206677], [0.0031600119546055794], [-0.13211898505687714], [0.06693486869335175], [-0.03826824575662613], [-0.05094216763973236]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_e53fbd5bc2ae1759131dd339dfdc4805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09301024675369263], [-0.03329252451658249], [-0.1414749026298523], [0.005552179645746946], [0.02272603288292885], [0.07824979722499847], [-0.10984474420547485], [0.02850295975804329], [0.04264805093407631]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.04462939873337746], [-0.09026241302490234], [-0.0763222873210907], [0.0036242941860109568], [0.025886045768857002], [-0.053869184106588364], [-0.042909879237413406], [-0.009765285067260265], [-0.008294115774333477]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_bc86432006fc7562fc58d539fa380ba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3cad1f8cbce2076a869387e82b80ccd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.00019229290774092078], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b980b93aaf428a718886917da12a05e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b980b93aaf428a718886917da12a05e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fab6118d64258223d34d611ecfe79ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(6165.861328125, dtype='float32').reshape([]),
            paddle.to_tensor([0.0018473155796527863], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_09ab8271e33b13dfa91bcd7116766d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(3912.12744140625, dtype='float32').reshape([]),
            paddle.to_tensor([0.0018473155796527863], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_8b5ee39a865a84dbddabba3af6b57e9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.01093385647982359, -0.020309938117861748, -0.0014530871994793415, 0.011614530347287655, -0.027645012363791466, -0.017154740169644356], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_183945d1705f589f0dc45699e5a47117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02786906436085701, 0.07007032632827759, 0.0267891027033329, 0.014066009782254696, 0.021699944511055946, 0.002614037599414587], dtype='float32').reshape([6]),
            paddle.to_tensor([0.008555368520319462, 0.1272517740726471, 0.0688125342130661, 0.022314582020044327, 0.22995047271251678, 0.07298903167247772], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_47482ed5c66a8843235ab6372ddd457a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.07253727316856384, 0.08720719814300537, -0.030573617666959763, -0.04275050759315491, 0.13802126049995422, 0.0461617112159729], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.1507343202829361, -0.23289291560649872, 0.0475274920463562, -0.27168169617652893, -0.2002953141927719, -0.4140487313270569], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_57f616a2faf1d81971295830500dd7d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.07804768532514572, 0.23803360760211945, 0.20417283475399017, -0.13806775212287903, -0.10154804587364197, 0.007378607988357544], dtype='float32').reshape([6]),
            paddle.to_tensor([0.04888084530830383, -0.03501257300376892, -0.08998474478721619, -0.22124311327934265, 0.41422635316848755, 0.2654237151145935], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_340137696cdf57b2c406059060009776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8636482954025269, 0.4609479010105133, 0.13824422657489777, 0.06544478237628937, 0.05339093878865242, 0.007810501381754875], dtype='float32').reshape([6]),
            paddle.to_tensor([1.8636482954025269, 1.460947871208191, 1.1382442712783813, 1.065444827079773, 1.0533909797668457, 1.0078104734420776], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d4e79e3c11042bc6870906613bb19a15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4e79e3c11042bc6870906613bb19a15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c5052d5d171e7576dd5195b25f8bd43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(15197.791015625, dtype='float32').reshape([]),
            paddle.to_tensor([0.16976314783096313], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3aef64781922969a92bd7f4a8a5b5923(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(102694.8515625, dtype='float32').reshape([]),
            paddle.to_tensor([0.16976314783096313], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8d36a51aa408ad29e965baf89bc4a40f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(950.7593994140625, dtype='float32').reshape([]),
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


class TestPrimitiveOp_8baf020901f83066619db6c41504519a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8baf020901f83066619db6c41504519a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ddf6e3ece57aceed9f609a1dc7026af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(452027.46875, dtype='float32').reshape([]),
            paddle.to_tensor([0.1837148666381836], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f628dc149ed7b422002b027d06726eae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(87539.9921875, dtype='float32').reshape([]),
            paddle.to_tensor([0.1837148666381836], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f483079d2241a8702dd70d8fdd8d3b5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b40b81719b4ae051a3ad01c76c48ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2480098158121109], [0.24436978995800018]]], dtype='float32').reshape([1, 2, 1]),
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


class TestPrimitiveOp_ae1f858b5ea33dd79ac3e20c5b1a0daf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.008202476426959038]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_25d71a6d93a822f3110e044f244c3777(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00041795894503593445]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.008620435371994972]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_ff4ebc068b94301212b9b4e14a6721ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.09975834935903549], [0.039119549095630646], [0.059811756014823914], [-0.10562919825315475], [-0.003041415009647608], [0.03176024928689003]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_3c77907ac2209e854ac7dbbe8159df28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11054292321205139], [-0.019648052752017975], [-0.05488264933228493], [0.10325108468532562], [0.0012183303479105234], [0.017783816903829575]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.01078457199037075], [0.01947149634361267], [0.004929106682538986], [-0.00237811217084527], [-0.0018230846617370844], [0.049544066190719604]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_9a283fb2eb39c8bcbc09e4b9c2388211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b40b81719b4ae051a3ad01c76c48ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24596427381038666]]], dtype='float32').reshape([1, 1, 1]),
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


class TestPrimitiveOp_b9fe13f5374667c109c4588c260c1156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(56.701995849609375, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9fca23ee73d8c6117a856385646a5809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(548.1932983398438, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_295862c5ccc03e945430f8a0d5a96c89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_295862c5ccc03e945430f8a0d5a96c89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b6774449152fbc81eabcbdfddfaa0d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(126918.4921875, dtype='float32').reshape([]),
            paddle.to_tensor([0.19278870522975922], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_616f36a6285fb817a519a09084c8abc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(117087.8828125, dtype='float32').reshape([]),
            paddle.to_tensor([0.19278870522975922], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_f675a1ed515fb525538ce9e9f8dc1a4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3cad1f8cbce2076a869387e82b80ccd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16409702599048615], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_845d89fd3fddf04c2db84dd653c1c3b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_845d89fd3fddf04c2db84dd653c1c3b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7571d76f6aeb76b07f2b342f3b9e355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-20477382.0, dtype='float32').reshape([]),
            paddle.to_tensor([0.321049302816391], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b290636a4b0ce3fcd928da5edd0366b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(264580.53125, dtype='float32').reshape([]),
            paddle.to_tensor([0.321049302816391], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eeaaf4213498cbdae47ae844b87bebae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_703b9c4a7e8742899c426a973f3f855f
    def get_inputs(self):
        return [
            paddle.to_tensor([303.8618469238281], dtype='float32').reshape([1]),
            paddle.to_tensor(2434.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_df8522fa6fd2890d3453bdb031377dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df8522fa6fd2890d3453bdb031377dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d1cd414bdc3bf60d46a8ef6e08a98f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(45306.5078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.03681889548897743], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_42a7dba11368f6baa5875b13c4563920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(14829.208984375, dtype='float32').reshape([]),
            paddle.to_tensor([0.03681889548897743], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_966a6d1466150b40b818848e45b7b2aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13051322102546692, 0.018631551414728165, 0.399068146944046, 0.4486573040485382], [0.038965631276369095, 0.44965460896492004, 0.4585685729980469, 0.125327467918396]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.3612508475780487, 0.3934182822704315, 0.07843542098999023, 0.4545356333255768], [0.10722312331199646, 0.2369028925895691, 0.0740278959274292, 0.35899680852890015]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_6ea4dd710e07c08e38d646acc8d667e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27385538816452026, 0.036572203040122986, 0.29701605439186096, 0.13144464790821075], [0.19793961942195892, 0.22078385949134827, 0.10303585976362228, 0.4376748204231262]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.07564617693424225, 0.42594537138938904, 0.2693103551864624, 0.31000953912734985], [0.4586943984031677, 0.3573634922504425, 0.2533925771713257, 0.250522255897522]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_e55525636f54c83701c7a57a52956f96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.022827766835689545], [0.007110994309186935], [0.06232514977455139], [-0.057350385934114456], [-0.041275642812252045]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_89f0467ed7d06c44bdfd38ec25e7ef28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02527928352355957], [-0.009511738084256649], [-0.05719303339719772], [0.09272189438343048], [0.05659174174070358]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0024515173863619566], [-0.00240074354223907], [0.005132114514708519], [0.035371504724025726], [0.015316097997128963]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_ec45210358c081ea3d50e30a5a2ada4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3cad1f8cbce2076a869387e82b80ccd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.44296854734420776], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_acb9d9fbf0d57db176f9691fef8c972e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acb9d9fbf0d57db176f9691fef8c972e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73b73fa82062be4508ce475c0e0c8106(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(274771.34375, dtype='float32').reshape([]),
            paddle.to_tensor([0.005685942247509956], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_864dfac3219d8662b80c8d4958f11dbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(134688.8125, dtype='float32').reshape([]),
            paddle.to_tensor([0.005685942247509956], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3038b40003fb3d6727f4f99c4f57e86f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3038b40003fb3d6727f4f99c4f57e86f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5727c6ca2dfc23e14eb21750e6ae3e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(55512.01171875, dtype='float32').reshape([]),
            paddle.to_tensor([0.35100457072257996], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b84aa31e988073e600894c6c9783d3f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(173631.03125, dtype='float32').reshape([]),
            paddle.to_tensor([0.35100457072257996], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_99a783021bae8f7cabf2021e17990277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99a783021bae8f7cabf2021e17990277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf524e62411d6859af4a5c78829ea358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-36836.5625, dtype='float32').reshape([]),
            paddle.to_tensor([0.31029972434043884], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_424963759c35ec030e093c33254e169f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(217033.890625, dtype='float32').reshape([]),
            paddle.to_tensor([0.31029972434043884], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f2578d91babc42d67a48dfdb4ca9fbd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3cad1f8cbce2076a869387e82b80ccd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.057580769062042236], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9fd93520018158bb64f4062d3c180d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c835d2ab69eea6d262b0284ffd198fb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(17.195253372192383, dtype='float32').reshape([]),
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


class TestPrimitiveOp_57e1dcdfb39e911a0ea7ffc298823634(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.014916401356458664], [0.027211245149374008], [0.014795346185564995], [0.052663881331682205]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_ef79f21762b7bb45a1d67bfa95e8faa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.053427569568157196], [0.16295942664146423], [0.05493427813053131], [0.10384605824947357]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.03851116821169853], [0.19017067551612854], [0.06972962617874146], [0.15650993585586548]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_92d7e41c6cdb7e0edacb6dce4fce546f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(4.292474746704102, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_48aa238594c8c624f177adf6d1a0eb60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48aa238594c8c624f177adf6d1a0eb60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d7841f8c7192ff873c7fac2063ad576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-397203.6875, dtype='float32').reshape([]),
            paddle.to_tensor([0.4510444402694702], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c69c70e4c72415991516ecf23daad65d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(29651.884765625, dtype='float32').reshape([]),
            paddle.to_tensor([0.4510444402694702], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5f1c6d95ecd345f44af03c9310b9bff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3cad1f8cbce2076a869387e82b80ccd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03847096115350723], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e956440e66f199ceeac653d0bab8ae28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(35.804141998291016, dtype='float32').reshape([]),
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


class TestPrimitiveOp_f2eb70ec74f85d255d3be35bee9690f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(237.85641479492188, dtype='float32').reshape([]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5933cde57c1487ce22b52f3ddce4d417(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(137.47593688964844, dtype='float32').reshape([]),
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


class TestPrimitiveOp_c3c7d689f29c400ee2799f3410dd1296(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3c7d689f29c400ee2799f3410dd1296(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e15c90a89304dc75f8ec95bb4291ca4
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b0cc39711d8342a91769bc5453f1192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(687381.75, dtype='float32').reshape([]),
            paddle.to_tensor([0.3597707152366638], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a1688ff1dd898767c194e94f2fd47596(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(239539.671875, dtype='float32').reshape([]),
            paddle.to_tensor([0.3597707152366638], dtype='float32').reshape([1]),
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