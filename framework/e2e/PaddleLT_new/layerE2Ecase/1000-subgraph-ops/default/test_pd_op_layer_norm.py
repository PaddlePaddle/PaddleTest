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



class PrimitiveOp_a2c49bb657f39b8742635fd454851d95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc6f3acb45fa0ae949fe57dc1a1aab9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c49bb657f39b8742635fd454851d95
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1cbe382a8db9d530acd98393b62b118(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a4c312a548fb5693a9f890a96d54375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1cbe382a8db9d530acd98393b62b118
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ee2d00feec4a1fb3454c6395c95878e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3402983af22243475683b169a6d35204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee2d00feec4a1fb3454c6395c95878e7
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2e5ba55a539c81817fd8148c9ac83143(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2800df3eaa3d00ea2a4328024bb4d2da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e5ba55a539c81817fd8148c9ac83143
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d969d2209b19c36614e3548d843925df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fcf3b869c9125a8efb503c3958d1b426(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d969d2209b19c36614e3548d843925df
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a57256a0f77028aa878f053883f7c088(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b12c7f9808c7993f9f36094ea446e51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a57256a0f77028aa878f053883f7c088
    def get_inputs(self):
        return [
            paddle.uniform([16, 1024, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c9c9aef95c01853031550be6cc2e875a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0321eac76f6d2c9eef2f242618c9cf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9c9aef95c01853031550be6cc2e875a
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c9445b07ce16cf80ead719a9efad6e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73739407f69875e56089a8dac9dcd865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c9445b07ce16cf80ead719a9efad6e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dcae3f8f3d59f3fec6beccb130c7c29e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d62af59667c1897057571e32404f25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcae3f8f3d59f3fec6beccb130c7c29e
    def get_inputs(self):
        return [
            paddle.uniform([128, 32, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_646f5dbb0eb37d92731e3b3bd0101935(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db285fe3e80d0aedb3981bb0685edc9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646f5dbb0eb37d92731e3b3bd0101935
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c4d0943cd58dd7d9333e67895129ad58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0b1fdef5a11dfa27dae1b1aee0833a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4d0943cd58dd7d9333e67895129ad58
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_000e898b0312425ea8b606a36e3d3816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1cbe382a8db9d530acd98393b62b118
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9f6223e6be42d3f73f1b69f38215cc0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_207697b9dedce7f9cabdca6be96e6f92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f6223e6be42d3f73f1b69f38215cc0e
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ff9cbeab9a1616e7c3b56daa379e9c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcae3f8f3d59f3fec6beccb130c7c29e
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7133f459258194713c2d9b615976d0e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e160ea3d47a4f328608c8491f4b1147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7133f459258194713c2d9b615976d0e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a8161109a1ca0500b857d7e4d46c9daf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9337e6131230dc409d58f255ccff321b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8161109a1ca0500b857d7e4d46c9daf
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c818bfbbef18edbfa5aa33ebb5e90895(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fda756c228a3cb89a6f25b0fbb8be63d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c818bfbbef18edbfa5aa33ebb5e90895
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ca88eb225fb31596a76203ed5cca078(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e98d950608ad22111848d24abf8c96e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ca88eb225fb31596a76203ed5cca078
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8f565b368774a26266d4cd236da054a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_30214ed16b4ee509c565cb0c6fd2efaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f565b368774a26266d4cd236da054a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_172d07f6f115fee3dffcc9ea09aa8251(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b6fc5e2c3d3197be7649e1001e57ba55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_172d07f6f115fee3dffcc9ea09aa8251
    def get_inputs(self):
        return [
            paddle.uniform([64, 256, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa7434c9067db69353b2a5b73b137b8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee2110e15361b50ace28a50aaed66d9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa7434c9067db69353b2a5b73b137b8a
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73739407f69875e56089a8dac9dcd865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c9445b07ce16cf80ead719a9efad6e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5d5accd78dec6778b1e02b2f911f9763(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ba63601670349d0028a1238081a11af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d5accd78dec6778b1e02b2f911f9763
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0b1fdef5a11dfa27dae1b1aee0833a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4d0943cd58dd7d9333e67895129ad58
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_22a3f090540d49a7d1d7b70024e0f776(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95b291a08369d9ea7051b6e777519d35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22a3f090540d49a7d1d7b70024e0f776
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c407c3184d352e62bef6a8a7d84b4f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a57256a0f77028aa878f053883f7c088
    def get_inputs(self):
        return [
            paddle.uniform([16, 512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01f5f0c774ca17ae6d108372d7e27882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_10e5e69ba93397c24650e841f937dd68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16384, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8b5620f9f00104fb3381bbc837809c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10e5e69ba93397c24650e841f937dd68
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dfb2242e760198ed4994bc6c8f9df0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8161109a1ca0500b857d7e4d46c9daf
    def get_inputs(self):
        return [
            paddle.uniform([8, 512, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db285fe3e80d0aedb3981bb0685edc9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646f5dbb0eb37d92731e3b3bd0101935
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27340c83ede84d41c21a35db2b7bf571(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_97c3ffb17bccfcb206de66750aed7b46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ff32b7eea39608d03c0bfe5a28ff6e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c3ffb17bccfcb206de66750aed7b46
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b78fecb155aac86f00ff5834e32f20a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee2d00feec4a1fb3454c6395c95878e7
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a7d8249a202c1b2eab7f950c28023b8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab5bc92830250cc1a9f573206d6b79f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d8249a202c1b2eab7f950c28023b8d
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4400be716d88f96559c532187ba4594a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e99ed28e07236ac62ccb90c09b839d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4400be716d88f96559c532187ba4594a
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e99ed28e07236ac62ccb90c09b839d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4400be716d88f96559c532187ba4594a
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7aab73e8592950743339b22409682cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646f5dbb0eb37d92731e3b3bd0101935
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c0394fb3b5096d247d212148760998f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bb35f6e867ea737f3c70e291770e9aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0394fb3b5096d247d212148760998f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5eef257ad690b3c1c09e19da10248967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fdbb56dd82d490a1fe850dfb155ec785(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df0a25e5bd903c07b399deb219fc0fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdbb56dd82d490a1fe850dfb155ec785
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_da0ff431a03a979dde55468e8fd64363(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65b0a0cb42c036b05492d879a39521c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da0ff431a03a979dde55468e8fd64363
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3a4c312a548fb5693a9f890a96d54375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1cbe382a8db9d530acd98393b62b118
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f84f925f17c9b124b3f72f08514d1bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646f5dbb0eb37d92731e3b3bd0101935
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01f5f0c774ca17ae6d108372d7e27882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3099c65b3e1fd2399be0deae57404dc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa7434c9067db69353b2a5b73b137b8a
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3099c65b3e1fd2399be0deae57404dc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa7434c9067db69353b2a5b73b137b8a
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbc3e099e4133af7d3fd9aed3ae38448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ca88eb225fb31596a76203ed5cca078
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01f5f0c774ca17ae6d108372d7e27882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af098c4845f0b42ddb7e9e5f5ff3d8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ca88eb225fb31596a76203ed5cca078
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e98d950608ad22111848d24abf8c96e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ca88eb225fb31596a76203ed5cca078
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2fcc3689cfababb60219456afa6a55c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_429868fb5ec71b0ede0a122d305fef61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fcc3689cfababb60219456afa6a55c4
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9f067b1643eefab2894f80523bd0798c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed0c96dde53acd56c0aaab9af14ba5a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f067b1643eefab2894f80523bd0798c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_93779a95e6da38c59f78cda85489af9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20faa3a83bd761c9f226f269291c01bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93779a95e6da38c59f78cda85489af9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_977e7a5846e4b26fb394d9ad76315e2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdbb56dd82d490a1fe850dfb155ec785
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5fbc94e7b2ebf44b0028e9029ec351b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4eea059113b5f3c67fb54e516fed7120(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fbc94e7b2ebf44b0028e9029ec351b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ece3ece2a271b9fc10e1b521de44001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_172d07f6f115fee3dffcc9ea09aa8251
    def get_inputs(self):
        return [
            paddle.uniform([16, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ee1130419c8753c01b18a1f16a5aa99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14393adc75d6f95ca65122a0a2763776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ee1130419c8753c01b18a1f16a5aa99
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc6f3acb45fa0ae949fe57dc1a1aab9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c49bb657f39b8742635fd454851d95
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9241568efebc25663ea9f9b205632833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee2d00feec4a1fb3454c6395c95878e7
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5f0ab6278cdd02043ab1d56faa28c7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0394fb3b5096d247d212148760998f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30214ed16b4ee509c565cb0c6fd2efaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f565b368774a26266d4cd236da054a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0dda30160aa4a43ace1c389c79a28396(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2560d0da5d26e6a311caed044959dd4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dda30160aa4a43ace1c389c79a28396
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_692a0d6985f15cca35e7de953889404c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ae451ca5f17519278a878095f276c60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692a0d6985f15cca35e7de953889404c
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d6e3574c7af29934c566adf0cf53dd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa7434c9067db69353b2a5b73b137b8a
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7aab73e8592950743339b22409682cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646f5dbb0eb37d92731e3b3bd0101935
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5ea97e82e09c5e44adcfab14ee7e3fb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b257231641d881af66d77fe9e3643cee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ea97e82e09c5e44adcfab14ee7e3fb2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5e7a121941abc01a3eb55439574d3920(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04a3028210bdf5944f2c567090f479f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e7a121941abc01a3eb55439574d3920
    def get_inputs(self):
        return [
            paddle.uniform([8, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e73007755ea638d3b33fd6c3fa34d7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ca88eb225fb31596a76203ed5cca078
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab5bc92830250cc1a9f573206d6b79f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7d8249a202c1b2eab7f950c28023b8d
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac075e71f4d0b872aefbead7efe3b450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_000e898b0312425ea8b606a36e3d3816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1cbe382a8db9d530acd98393b62b118
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95b291a08369d9ea7051b6e777519d35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22a3f090540d49a7d1d7b70024e0f776
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c78138fecc17551870394343d0a08e7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc4505c5f0453b70172a0c0498aa6ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c78138fecc17551870394343d0a08e7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2843185fdff1e12c671d0fc18cd0c0e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c3ffb17bccfcb206de66750aed7b46
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_496fec7c817f3f4868fcc4dd09e832cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646f5dbb0eb37d92731e3b3bd0101935
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7aab73e8592950743339b22409682cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646f5dbb0eb37d92731e3b3bd0101935
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3c6e015ecae233ea2515b2c0f2ca03e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b61c88e8eef52c4ad8a5d0fec18e3e68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c6e015ecae233ea2515b2c0f2ca03e6
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45a60579a492a653122aea7fb38f49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3099c65b3e1fd2399be0deae57404dc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa7434c9067db69353b2a5b73b137b8a
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_207697b9dedce7f9cabdca6be96e6f92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f6223e6be42d3f73f1b69f38215cc0e
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc378c79fce9a31c655a2c8782e6e20d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d5accd78dec6778b1e02b2f911f9763
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d4629b4898f409df73554f79dc084c35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a659eb51b7456d1767d0aad33c2295de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4629b4898f409df73554f79dc084c35
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ee83768113c64ef8bd5df0bd15730bb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1efbc0a93dad477defb528fdafd9eb07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee83768113c64ef8bd5df0bd15730bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09219792e133f4d5cb7750fa6855af25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e73007755ea638d3b33fd6c3fa34d7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ca88eb225fb31596a76203ed5cca078
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e98d950608ad22111848d24abf8c96e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ca88eb225fb31596a76203ed5cca078
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09219792e133f4d5cb7750fa6855af25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fae362556a5aa035487dce30da1344ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646f5dbb0eb37d92731e3b3bd0101935
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db285fe3e80d0aedb3981bb0685edc9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646f5dbb0eb37d92731e3b3bd0101935
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac075e71f4d0b872aefbead7efe3b450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e98d950608ad22111848d24abf8c96e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ca88eb225fb31596a76203ed5cca078
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_402421c8b0dc3b27d74b7fab1c60b517(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_895c777ce06244c3b540b5093cc9b78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_402421c8b0dc3b27d74b7fab1c60b517
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db285fe3e80d0aedb3981bb0685edc9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646f5dbb0eb37d92731e3b3bd0101935
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5eef257ad690b3c1c09e19da10248967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7aab73e8592950743339b22409682cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646f5dbb0eb37d92731e3b3bd0101935
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_001ae87713896c37841662055399c903(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bf6c7ee819ec91c39c310c16b6d576b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_001ae87713896c37841662055399c903
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb0c0e8b32baabcb2afe4197cb5999fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee83768113c64ef8bd5df0bd15730bb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b257231641d881af66d77fe9e3643cee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ea97e82e09c5e44adcfab14ee7e3fb2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3b72dda2c0cd0a7f8b5d05b4798000e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93779a95e6da38c59f78cda85489af9f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0321eac76f6d2c9eef2f242618c9cf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9c9aef95c01853031550be6cc2e875a
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd1f4fb101cd2ea7eff7a38a22d571bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_402421c8b0dc3b27d74b7fab1c60b517
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee2110e15361b50ace28a50aaed66d9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa7434c9067db69353b2a5b73b137b8a
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1bc48d9062d1fa333d5463df0f52225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa7434c9067db69353b2a5b73b137b8a
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2800df3eaa3d00ea2a4328024bb4d2da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e5ba55a539c81817fd8148c9ac83143
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e73007755ea638d3b33fd6c3fa34d7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ca88eb225fb31596a76203ed5cca078
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65b0a0cb42c036b05492d879a39521c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da0ff431a03a979dde55468e8fd64363
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee2110e15361b50ace28a50aaed66d9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa7434c9067db69353b2a5b73b137b8a
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01f5f0c774ca17ae6d108372d7e27882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84c54acadbca3021dc8d2138247caf60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ca88eb225fb31596a76203ed5cca078
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8b5620f9f00104fb3381bbc837809c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10e5e69ba93397c24650e841f937dd68
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96ca778a1317c6a29979d42d2bb27e1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c78138fecc17551870394343d0a08e7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e160ea3d47a4f328608c8491f4b1147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7133f459258194713c2d9b615976d0e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc4505c5f0453b70172a0c0498aa6ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c78138fecc17551870394343d0a08e7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1bf6c7ee819ec91c39c310c16b6d576b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_001ae87713896c37841662055399c903
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8086d815ccc88fdb8856609174c93279(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef647d83b60bc37543b1fa28317e698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8086d815ccc88fdb8856609174c93279
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80357abc1d849e48850c4fde921a15bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee2d00feec4a1fb3454c6395c95878e7
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4eea059113b5f3c67fb54e516fed7120(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fbc94e7b2ebf44b0028e9029ec351b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c42374ff7fc6c71c1e6ead5f21cc6b4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dda30160aa4a43ace1c389c79a28396
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a659eb51b7456d1767d0aad33c2295de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4629b4898f409df73554f79dc084c35
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b72ef7319bfce5983b824975b9060e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8086d815ccc88fdb8856609174c93279
    def get_inputs(self):
        return [
            paddle.uniform([4, 128, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e73007755ea638d3b33fd6c3fa34d7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ca88eb225fb31596a76203ed5cca078
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_317b1a9810fb50234c6bda155aea2e08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdbb56dd82d490a1fe850dfb155ec785
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a01207db160a249ae22895fcc4d64082(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ca88eb225fb31596a76203ed5cca078
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed0c96dde53acd56c0aaab9af14ba5a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f067b1643eefab2894f80523bd0798c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1b6518278918549d4bd2cfc353dad4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e7a121941abc01a3eb55439574d3920
    def get_inputs(self):
        return [
            paddle.uniform([4, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6779bbd947b8052290894b4d540b727d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_646f5dbb0eb37d92731e3b3bd0101935
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcf3b869c9125a8efb503c3958d1b426(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d969d2209b19c36614e3548d843925df
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96ca778a1317c6a29979d42d2bb27e1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c78138fecc17551870394343d0a08e7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac075e71f4d0b872aefbead7efe3b450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac075e71f4d0b872aefbead7efe3b450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3e8a9ba52a0ea24b97d95f6a4e3ac4
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20ea36c95ef54aa44bc2e64122342cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdbb56dd82d490a1fe850dfb155ec785
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()