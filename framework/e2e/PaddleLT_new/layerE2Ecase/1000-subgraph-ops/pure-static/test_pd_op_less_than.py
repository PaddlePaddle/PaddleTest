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


class TestPrimitiveOp_be73832a1ac30153790cb03fcc705af4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d764e45b90fea95a0f95a560835c65f
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
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


class TestPrimitiveOp_8734825683bff55688a3c56c5e97e4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4046d6522e82a4d349f3b5ca651768f7
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_6e8b7deb48636306aaae0266d05ee9b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_918abaf87fe604a44282f3618cd211a6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
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


class TestPrimitiveOp_dd8794c516d7e3edd58e08529206dca1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a600901b7a25416c46857679f180f9c8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class PrimitiveOp_84584ca190b566b9822fa9f6b4fa1bda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_192d016ebbe405a341d56bf0e25e15c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84584ca190b566b9822fa9f6b4fa1bda
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1e24080b4192d4620d2c318bc2359a60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_918abaf87fe604a44282f3618cd211a6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


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


class TestPrimitiveOp_54117a17704b7ec93fab66a6443cf5d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96f82dd8c578dff8fa5cde03224354e1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_fe8829c4c924b97448963f9625caf38c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96f82dd8c578dff8fa5cde03224354e1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
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


class TestPrimitiveOp_bb99d6a793b5588b4ca29ce17000582f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_584f603a3f202f87255fb61de882ab5c
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a73677904ae071b5ea6273640195df8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_950462dcc85dfd36a32101c2273024ae
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
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


class TestPrimitiveOp_28997bb06d489d74828edd78278e3575(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a43a24b1033799be0c1e7cab68da36f1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
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


class TestPrimitiveOp_cb60547e006820a50b109f3bbba8bd5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_729c05f9d29ec1baae32347c012aa7f4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
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


class TestPrimitiveOp_e9510944c06a547f138d23ef66b69076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8bf0904007e6c74347d1a0dfeee2e21
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
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


class TestPrimitiveOp_bd936aeb7a68811e6122105b846a7ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5634ddbcf083b79ae6a019cddc6debf
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
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


class TestPrimitiveOp_661821cadd866040afe6664be5a39b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bdb7adb91f7925f21a11d18efb49b88
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
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


class TestPrimitiveOp_f3e2d2017df851265164b2b53e4f5d72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbf3de6078cd2abd0fb3c46505cee36b
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
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


class TestPrimitiveOp_ac8cf8ca78dae594125c6d590b46a4c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_946a287e3db6b9057bc86a519d60cf13
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dbb2e2bf68f18f40b6822de78e43461d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8bf0904007e6c74347d1a0dfeee2e21
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_6e8b7deb48636306aaae0266d05ee9b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_918abaf87fe604a44282f3618cd211a6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_f0cba51820f129682d7df745afbe4f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5634ddbcf083b79ae6a019cddc6debf
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3558c057ac09c321eac538088f2d9fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a43a24b1033799be0c1e7cab68da36f1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()