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



class PrimitiveOp_96f82dd8c578dff8fa5cde03224354e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f45d6bf2e92d877f25aaff0a1da5c0fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96f82dd8c578dff8fa5cde03224354e1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[15200], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class PrimitiveOp_584f603a3f202f87255fb61de882ab5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5e37ddbbc581d0b558f70c7900174d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_584f603a3f202f87255fb61de882ab5c
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class PrimitiveOp_918abaf87fe604a44282f3618cd211a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fde04c6ed42c8ce16e4d66232b7e370a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_918abaf87fe604a44282f3618cd211a6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_fde04c6ed42c8ce16e4d66232b7e370a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_918abaf87fe604a44282f3618cd211a6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class PrimitiveOp_950462dcc85dfd36a32101c2273024ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c097b6fe33e10803813425853247ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_950462dcc85dfd36a32101c2273024ae
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2204], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class PrimitiveOp_f5634ddbcf083b79ae6a019cddc6debf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0aafbd25a745002cf8b02c0878e9240f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5634ddbcf083b79ae6a019cddc6debf
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[950], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_0b7745fc2312911ca3704cf9685afcdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96f82dd8c578dff8fa5cde03224354e1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[15200], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class PrimitiveOp_3bdb7adb91f7925f21a11d18efb49b88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9a2f5d9a97162d4e069605c1e405d91a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bdb7adb91f7925f21a11d18efb49b88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[8816], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class PrimitiveOp_a600901b7a25416c46857679f180f9c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24c5064075df3b4aba5f570731f022bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a600901b7a25416c46857679f180f9c8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[150], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class PrimitiveOp_a43a24b1033799be0c1e7cab68da36f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a40ef3acf9a539ec31d4591c6722dcb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a43a24b1033799be0c1e7cab68da36f1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[70], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class PrimitiveOp_946a287e3db6b9057bc86a519d60cf13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e646c268a5142a6f2feb1d08fdd4953(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_946a287e3db6b9057bc86a519d60cf13
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9068b83c9493a20da6ea3d812db02efc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5634ddbcf083b79ae6a019cddc6debf
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[950], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class PrimitiveOp_fbf3de6078cd2abd0fb3c46505cee36b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8c41cf13952377d0686acbb2b89acf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf3de6078cd2abd0fb3c46505cee36b
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_91b56563a0ed656f224c9ccc373e9778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a43a24b1033799be0c1e7cab68da36f1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[70], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class PrimitiveOp_729c05f9d29ec1baae32347c012aa7f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6233d2cd0e01f77e6dafc8f5344e3090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_729c05f9d29ec1baae32347c012aa7f4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[551], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class PrimitiveOp_4046d6522e82a4d349f3b5ca651768f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67d852ff2337cf372326d154456277f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4046d6522e82a4d349f3b5ca651768f7
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_faa971e55be40cb137738ea08c0b6891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_918abaf87fe604a44282f3618cd211a6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3800], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class PrimitiveOp_6d764e45b90fea95a0f95a560835c65f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_256fe1a1c5550e4f38dd4055c38c8509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d764e45b90fea95a0f95a560835c65f
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e8bf0904007e6c74347d1a0dfeee2e21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c4d98840f8c18668969db10b3ae1806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8bf0904007e6c74347d1a0dfeee2e21
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[247], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_f864b748ec5b709e610e17636599708e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8bf0904007e6c74347d1a0dfeee2e21
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[247], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()