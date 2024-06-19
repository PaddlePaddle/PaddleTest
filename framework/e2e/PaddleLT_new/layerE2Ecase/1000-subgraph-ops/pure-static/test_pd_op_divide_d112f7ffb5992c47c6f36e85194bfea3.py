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



class PrimitiveOp_25af0ca8b1099f1a176e9d4b52013b77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0e72f83f9a7fae8cffbe2dc5f373e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25af0ca8b1099f1a176e9d4b52013b77
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e7b11a5036204283a2c5d8f1bb141f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(18496.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1c7a4be3a8fbfed1cba7577260102b48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_892e1ccd7414c6585e5ce259ae1abb49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c7a4be3a8fbfed1cba7577260102b48
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_61a805987e1e8cb6a172636260758ae4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 21824], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a42db9a4015c5ca34fc459768c6272a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[0.0005724666989408433], [-0.01464309636503458], [-0.03977613523602486], [-0.0020404267124831676], [-0.016553107649087906], [-0.10037871450185776]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_2748e23cb717ffee08ff8f19c340129e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[0.01920875534415245], [-0.05448722839355469], [-0.7929041385650635], [-0.025595396757125854], [-0.09671010822057724], [-0.0031508521642535925]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_042534739d9da3309d432ade46c5f7ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61a805987e1e8cb6a172636260758ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[0.06838793307542801], [-0.19067531824111938], [-0.2141905128955841], [-0.16766908764839172], [-0.13806064426898956], [-0.16420939564704895]]], dtype='float32').reshape([1, 6, 1]),
        ]


class PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c091cbba25152ff2749cf59bff8fda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9c091cbba25152ff2749cf59bff8fda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5f05a1746a091fe779f485a29168fe33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 100, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_799fd4483319a2fa7e336cbaf1de6ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f05a1746a091fe779f485a29168fe33
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e7a7e07c98df6632d59175511f8874a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d84e1f1ec383e37c9619b37de1cd676e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7a7e07c98df6632d59175511f8874a8
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_148d6ac01daccbf3ef5985d4a721ccdc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfc042859f51ae18383a6f7b90c27a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_148d6ac01daccbf3ef5985d4a721ccdc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3122318387031555, 0.18198776245117188, -0.26129961013793945, 0.45018064975738525], [0.17244255542755127, -0.32755178213119507, -0.3703860342502594, -0.42434749007225037]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[-0.26298946142196655, -0.0028013288974761963, 0.3600921630859375, 0.3126152753829956], [-0.4940076470375061, 0.28963011503219604, -0.010319530963897705, -0.25493645668029785]], dtype='float32').reshape([2, 4]),
        ]


class PrimitiveOp_6381ac9adbaa89dec0c7594136501af0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13efa6dba8e367cebda8c84b7eba7347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6381ac9adbaa89dec0c7594136501af0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_13efa6dba8e367cebda8c84b7eba7347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6381ac9adbaa89dec0c7594136501af0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_41724c6d1dc408f2518b7c352b7d5d2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 1025, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b464121430e7ec9a271cb57e8f464984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41724c6d1dc408f2518b7c352b7d5d2f
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52371865f0b97aa2991946fd00248ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2050.1650390625, dtype='float32').reshape([]),
            paddle.to_tensor(6168.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9d2c1e94443db3c059ce7dadaca5ba5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1542, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d467e0593f6a98e6973fbbf7668d5338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d2c1e94443db3c059ce7dadaca5ba5d
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d467e0593f6a98e6973fbbf7668d5338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d2c1e94443db3c059ce7dadaca5ba5d
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d9b6439532c9ba79072152eb89e60f87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ddc44d51ebbc8a44e809d6c6cf02b4dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-92272.53125, dtype='float32').reshape([]),
            paddle.to_tensor([0.4371209740638733], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_744d4ebb37ac9ac5e479de8bfd6b7d54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c56e66449081cc4fe01b56fe8e49582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_744d4ebb37ac9ac5e479de8bfd6b7d54
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0cd9e88be41433cdff7466410b66f43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-179.30813598632812, dtype='float32').reshape([]),
            paddle.to_tensor([0.4371209740638733], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_339033ebf449a1e0d58429e4223ebeb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(3124.5322265625, dtype='float32').reshape([]),
            paddle.to_tensor(9444.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_74e79a0df238750941efa6f755a6542d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2361, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef575b4fe021cd1fb8c1fccbbee1caf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74e79a0df238750941efa6f755a6542d
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef575b4fe021cd1fb8c1fccbbee1caf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74e79a0df238750941efa6f755a6542d
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e87a7b234908dac8de37c3770750f755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-7089.873046875, dtype='float32').reshape([]),
            paddle.to_tensor([-0.19258850812911987], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_95ab3c48ba122237a88c964c5bf992d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_933d43604c92fb8c60d02ebfb006d015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95ab3c48ba122237a88c964c5bf992d2
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ebea29c50d3da4f8bf68379705812d3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-76.39175415039062, dtype='float32').reshape([]),
            paddle.to_tensor([-0.19258850812911987], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_eceeae894259106eef06544586420874(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d669c0581e5c6cc305dbad3faf62b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eceeae894259106eef06544586420874
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(144.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_241011960a8cb233f88498b30179eeff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f5c8c1a6d1b86e4e3b031b5618908c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_241011960a8cb233f88498b30179eeff
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8f5c8c1a6d1b86e4e3b031b5618908c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_241011960a8cb233f88498b30179eeff
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_61592fc5fc967acf1cfbd8b026ad8840(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 1024, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9b57a86948d003bf58bde8b53b06f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61592fc5fc967acf1cfbd8b026ad8840
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_91f56cb14bb241275d35705ca5453495(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 2304, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1db042ac0604c07ad0c3217c4909edea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91f56cb14bb241275d35705ca5453495
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1db042ac0604c07ad0c3217c4909edea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91f56cb14bb241275d35705ca5453495
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_98d9d8acbdbbc609aaec5592219aeced(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1096347d31a41a9338de90fe4a716aea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98d9d8acbdbbc609aaec5592219aeced
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(14400.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a47274817015a7349f92259db4706444(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_123458163592094d66722fa8aadcafd9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25f47c97a4036b866c4becaa9afd2981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123458163592094d66722fa8aadcafd9
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25f47c97a4036b866c4becaa9afd2981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123458163592094d66722fa8aadcafd9
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_78b9fbfda75fc362683e18e110f8a74c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a69e06d82f440dfa469849299ae89096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78b9fbfda75fc362683e18e110f8a74c
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a69e06d82f440dfa469849299ae89096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78b9fbfda75fc362683e18e110f8a74c
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9bed0a8c1232d259af5d750ac1058037(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd47e633ed68fe47dc56876f4358d2db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bed0a8c1232d259af5d750ac1058037
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dd47e633ed68fe47dc56876f4358d2db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bed0a8c1232d259af5d750ac1058037
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_200e6629d14ef72ab61bc26745d6b3f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 1, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8617ddb79b89aa74ff1dae2373aedbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_200e6629d14ef72ab61bc26745d6b3f2
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a8617ddb79b89aa74ff1dae2373aedbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_200e6629d14ef72ab61bc26745d6b3f2
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e13cdb09e433b59dd353aef918254000(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ba0606e889c830d669000f7179fa084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1600.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b1f39f603d43ab4921d299a6df37b647(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afae707c7993638537d26131333dde98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1f39f603d43ab4921d299a6df37b647
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_25faaed29840855748cd16df4fb2bdab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e1c0c171895aededff7cb4441b55797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25faaed29840855748cd16df4fb2bdab
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8e1c0c171895aededff7cb4441b55797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25faaed29840855748cd16df4fb2bdab
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a68da5c2e8d7ad3c6c4635e4fcc10bc1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 197, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e98db9450aeafb2834e115abefe6cdef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a68da5c2e8d7ad3c6c4635e4fcc10bc1
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2468c61eef5edcc5d7aedb9ef8aeab54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2468c61eef5edcc5d7aedb9ef8aeab54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_64508d0b7fb630d6b59223975ab7586e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf9656c4d3892200f291ae0fb4aeafb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64508d0b7fb630d6b59223975ab7586e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.11927862465381622], [-0.0435338169336319], [-0.15602411329746246], [-0.2382054328918457]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_195e499db5aeaacd635093f61793f833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64508d0b7fb630d6b59223975ab7586e
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30871984362602234], [0.04665578529238701], [0.026612654328346252], [0.4145473539829254]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.18944121897220612], [0.003121968824416399], [-0.1294114589691162], [0.1763419210910797]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_61789ba15ebba94914e8fbae29cdbfef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-2.6904590129852295, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5c4dac6ec2f7998539fe40798a95b24f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 7581, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f3074acb96bd631907752b429a6af81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c4dac6ec2f7998539fe40798a95b24f
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 7581, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73ec8b8b7db535b47dfb37b3341da0d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, -0.0, 0.0, -0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.06958289444446564, 0.036181844770908356, -0.004434196278452873, 0.0199829563498497, 0.0050819143652915955, 0.014829179272055626], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_0aebdd3d7546bab168e5647cc6a64a95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1405324637889862, 0.3278995156288147, 0.5568813681602478, 0.3690003752708435, 0.2866296172142029, 0.5118327736854553], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3210272192955017, 1.2097169160842896, 0.5724218487739563, 0.9801535606384277, 0.12928898632526398, 0.6481572389602661], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_8a3e613febd639204aae09e10e64d8ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.31115055084228516, 0.04071506857872009, -0.15438109636306763, 0.6036527156829834, -0.018606901168823242, -0.13000443577766418], dtype='float32').reshape([6]),
            paddle.to_tensor([0.22363096475601196, 0.8886597752571106, 0.028722405433654785, -0.08792230486869812, -0.27311986684799194, -0.1140667200088501], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ac10cee4b883a055902cbaaeb36b036b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.38973838090896606, 0.13094156980514526, 0.1581231951713562, 0.45304128527641296, -0.3166550397872925, 0.5811730623245239], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.16987508535385132, -0.7427990436553955, -0.02934253215789795, 0.16126012802124023, 0.074748694896698, -0.22679078578948975], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_3bf45ba1403cee623edd84c41dacdaa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39c526fd111fcd8d7384d75c9d26791
    def get_inputs(self):
        return [
            paddle.to_tensor([1.7998747825622559, 0.019664479419589043, 8.764610726075261e-08, 2.8568501472473145, 0.8023276925086975, 1.702123999595642], dtype='float32').reshape([6]),
            paddle.to_tensor([2.799874782562256, 1.0196645259857178, 1.0000001192092896, 3.8568501472473145, 1.8023276329040527, 2.7021241188049316], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b4a350ae5eecb3c727ae388634b8531d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(15.76768970489502, dtype='float32').reshape([]),
            paddle.to_tensor(6.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_62a255c61caad2c4531435061d42647e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2100, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8dcf9a8b9ea398deb796b2b9623ee020(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62a255c61caad2c4531435061d42647e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2100, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4b926f9a55a654e841e6244fd3d5fe3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-4.771322250366211, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9f209460593495992a15bc5caa65ec1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87ffc4335e9ea34beef51f6851e16fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f209460593495992a15bc5caa65ec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(28224.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b77adfe5025993ea63c577e050b8e2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b77adfe5025993ea63c577e050b8e2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d950b5f47cc8adf9a94f191208296729(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1db07e8367d70f64323e92a12f2340d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d950b5f47cc8adf9a94f191208296729
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 50, 76], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e74f60fee84f5eb01971fec659900062(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fb059da76e2d04c6a77999c45ae78ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e74f60fee84f5eb01971fec659900062
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_6a984ad88923d9ce9e3ceb8bc666c394(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3536cfa125c1fc643ad6a70bda86c5cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a984ad88923d9ce9e3ceb8bc666c394
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3536cfa125c1fc643ad6a70bda86c5cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a984ad88923d9ce9e3ceb8bc666c394
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f56409e3719d737e0ff60edf2543900d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 640], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 640, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c16820b7f19fe75ffad6b58fd5ab8029(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f56409e3719d737e0ff60edf2543900d
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 640, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_664ee626c9a7bc570e987b6e1ded6df1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98d9d8acbdbbc609aaec5592219aeced
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(10816.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b77adfe5025993ea63c577e050b8e2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b77adfe5025993ea63c577e050b8e2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1aaaa64cbdaef9d69fa69ac5497e7945(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ebc7db99b1d61dec15d7e951f04183f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aaaa64cbdaef9d69fa69ac5497e7945
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ad6ecad809d350691b47c24d8a26d762(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df8207840e16cfbb4239d49254255314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad6ecad809d350691b47c24d8a26d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae95baf46c66a2712f8e813f17f7f0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25af0ca8b1099f1a176e9d4b52013b77
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(169.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_2ca4e5ab2c2ee85c9b5fa42e6272cb60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fbdad52a5a49066b17727e500bda521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ca4e5ab2c2ee85c9b5fa42e6272cb60
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d3daeeaa5f905c5b4aa94d8490adb6c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_148d6ac01daccbf3ef5985d4a721ccdc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.35356512665748596, 0.405293345451355, 0.46430832147598267, -0.26953333616256714], [0.3279486298561096, -0.19524812698364258, -0.49333328008651733, 0.3488161563873291]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.33140259981155396, 0.279971182346344, 0.0695924162864685, 0.4702419638633728], [-0.10671475529670715, -0.23206189274787903, 0.32643330097198486, 0.4896619915962219]], dtype='float32').reshape([2, 4]),
        ]


class PrimitiveOp_9765427904702b3cec29af184420b13e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5c68226d73ea19ad96745c18a7052b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2704.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b52152804f4242ccda695798e1a24ed3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d63dfa7725a4d6ef6a3ebc6e92664e0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b52152804f4242ccda695798e1a24ed3
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 100, 152], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_798db199564304f5cd70c6a68423a2c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a41c89fe1695cce953362f02cfabce12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_798db199564304f5cd70c6a68423a2c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_07078f5e631eff26eba52f1d8b40799b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21, 16384], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5aa270d809716d1393f337d807c63d87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07078f5e631eff26eba52f1d8b40799b
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[10356.4736328125], [10400.4921875], [10369.83203125], [10386.8193359375], [10342.953125], [10370.44140625], [10386.9892578125], [10381.8359375], [10350.208984375], [10373.75], [10362.9375], [10331.630859375], [10346.904296875], [10315.0458984375], [10354.1123046875], [10370.546875], [10392.298828125], [10340.076171875], [10345.8701171875], [10373.333984375], [10371.080078125]]], dtype='float32').reshape([1, 21, 1]),
        ]


class PrimitiveOp_849ec7c645ad4a347bd586ebdbc99fcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9c4143af5c5597fcc1f13bbfeb25a29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_849ec7c645ad4a347bd586ebdbc99fcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e991b1e1d68b589453cda60b5040c397(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afb18cb965c43c507f85f1b287bc864f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e991b1e1d68b589453cda60b5040c397
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_afb18cb965c43c507f85f1b287bc864f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e991b1e1d68b589453cda60b5040c397
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ecd5aa0821478e2032e0886ad088d9e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 200, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31d259fdbdd774ec53ef8e915bb3ca36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecd5aa0821478e2032e0886ad088d9e8
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 200, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_324a042b2e220af09c5477a86aeb91d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_324a042b2e220af09c5477a86aeb91d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_71acdc1388cce56742292403d11b605d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4096.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_8357bb4e351fddef99148117a3fc5989(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_044441e54092dfcc67ece786805263be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8357bb4e351fddef99148117a3fc5989
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.041902512311935425], [-0.4516246020793915], [-0.012981231324374676], [0.33478492498397827], [0.12920251488685608], [-0.06198274716734886]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_8bca406f8c7daa00e7ad15d02a7159af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8357bb4e351fddef99148117a3fc5989
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1343022882938385], [0.7727550864219666], [0.01156703196465969], [-0.17761173844337463], [-0.21529260277748108], [0.055835969746112823]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.17620480060577393], [0.3211304843425751], [-0.0014141988940536976], [0.15717318654060364], [-0.086090087890625], [-0.006146775558590889]], dtype='float32').reshape([6, 1]),
        ]


class PrimitiveOp_89faa8f9cfceb6a68c0b564dd96973be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9261, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc45fd7f184ed5b47a89609c5e0ad127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89faa8f9cfceb6a68c0b564dd96973be
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9261, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_48b9aa3cd446aa0bbd2be50fb70662d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4116, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_025a0af5aa7f376da763cc6ff7c000fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48b9aa3cd446aa0bbd2be50fb70662d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c86e7ae0c322e377577b1a83148500f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1600.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2794f233f52bd7c24e3e81875da463e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2794f233f52bd7c24e3e81875da463e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d218101ae6cc2c945447d4e5f3be809b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b563b0359625a6bd32da4f58fcefda06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d218101ae6cc2c945447d4e5f3be809b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 7, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_608122f3fc89908f538ec0234f0b8d78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e927a4ddbcfcaf0aee587ea9125bc47d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_608122f3fc89908f538ec0234f0b8d78
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ac05ad983416ae2ba1bf1d5a370574f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(400.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_04d88777bb5cb50fb086f6eb864923e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(676.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d20301e725e12d2a28055c2e7c0e5243(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 60800, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4cb5e0d7f3a7543cbb70c92d31ffa030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d20301e725e12d2a28055c2e7c0e5243
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4cb5e0d7f3a7543cbb70c92d31ffa030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d20301e725e12d2a28055c2e7c0e5243
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4c93e703bb68b6e2ca766cb0487d6aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4c93e703bb68b6e2ca766cb0487d6aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_eb1495e94fb16d27855af6b6599266b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf4c9ff9eb35790557ff2378ce0d8749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb1495e94fb16d27855af6b6599266b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_048890a093f8fae44ee60cdb2d2208ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_048890a093f8fae44ee60cdb2d2208ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_353035a67891e73e0688bf38c37d5a28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21504, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_061ee16774935beeafa8fccdd80972a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_353035a67891e73e0688bf38c37d5a28
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21504, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_471753e5a1f43b1877e5fa9a2038ad6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 9216, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfd3598d0cfa0438d6afb30cd7f346e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471753e5a1f43b1877e5fa9a2038ad6e
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bfd3598d0cfa0438d6afb30cd7f346e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471753e5a1f43b1877e5fa9a2038ad6e
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f5519ba142a258f164da420bf7b4c1d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e3b3875a7f8c285245f5a8392b6a968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5519ba142a258f164da420bf7b4c1d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(441.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_3f63516d98502bf921317be94db2215c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_210b6b79c731e4aba93f558e9047993b
    def get_inputs(self):
        return [
            paddle.uniform([20267, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([20267, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_76e2bd3d4a1e4c8208b41a22bd34f984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2116.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_590e4992a892870d7216041d70dc87b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(1.8602583408355713, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6db9cbb824dce374882339b92a19e25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98d9d8acbdbbc609aaec5592219aeced
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(65536.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_39d9ef8c2243d1b1e1d36f72628e322b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1ef068be5ba98d41d2ca6bfef06af9a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d9ef8c2243d1b1e1d36f72628e322b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.2701488137245178], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_900e53729608569f217c410830f2681b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(6400.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9288920304028b1ae986841ae437f48b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cdfff13bd38db149b8bec0edba751f69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9288920304028b1ae986841ae437f48b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ce8e564d8db99a9774470bf0ff128f93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(169.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_6788cd52d4620f93775337d648c1c774(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cccd621cc3421138f9edabf34494595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6788cd52d4620f93775337d648c1c774
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9cccd621cc3421138f9edabf34494595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6788cd52d4620f93775337d648c1c774
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1c67356afeef3c6662fa69cb09d4b9e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 160, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9412f50e90b6e3817b4d776a383f4d7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c67356afeef3c6662fa69cb09d4b9e1
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_293e05703d68ce7a3d9e5544460dfa48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eceeae894259106eef06544586420874
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d76b5ec1e1c4784a6ec33688d260f4c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a335037e6b6da03dd192d43505998719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76b5ec1e1c4784a6ec33688d260f4c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_259a7c6c025cc531f64f7e4687c7ab22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85b1700baa4d50eb732de4f8ba1ffd35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_259a7c6c025cc531f64f7e4687c7ab22
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_85b1700baa4d50eb732de4f8ba1ffd35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_259a7c6c025cc531f64f7e4687c7ab22
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_85b1700baa4d50eb732de4f8ba1ffd35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_259a7c6c025cc531f64f7e4687c7ab22
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_85b1700baa4d50eb732de4f8ba1ffd35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_259a7c6c025cc531f64f7e4687c7ab22
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3db0c9d579a52992e133a58e987f65d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2753.128662109375, dtype='float32').reshape([]),
            paddle.to_tensor(8212.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_4f1c6b8c381b27defde071c39a01ad05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2053, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62a21987374d887c0a1db91be4db4a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f1c6b8c381b27defde071c39a01ad05
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_62a21987374d887c0a1db91be4db4a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f1c6b8c381b27defde071c39a01ad05
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1b236786b45f6a2b63c6447022b6f010(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-14643.224609375, dtype='float32').reshape([]),
            paddle.to_tensor([0.1699134111404419], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1172a340e650611a436756ebc7b9f086(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_940248fec8025f15dca9070a7d27b8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1172a340e650611a436756ebc7b9f086
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3df98215ec3298ac1e438dacc56ba0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-444.95196533203125, dtype='float32').reshape([]),
            paddle.to_tensor([0.1699134111404419], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_13ef74aa9f50b42fe3d36a1229797140(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cff609ee882a2a654110d543d9fd4b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ef74aa9f50b42fe3d36a1229797140
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ca25a97a8819c2aac095142f2b4ff7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eceeae894259106eef06544586420874
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(529.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_92701321a6b7e6725c5d1a71eb0efc27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de86c7793f74724dd11127e1f11b852a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92701321a6b7e6725c5d1a71eb0efc27
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(32.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_de86c7793f74724dd11127e1f11b852a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92701321a6b7e6725c5d1a71eb0efc27
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(32.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8032a7476c099fb04877e5b6bce55d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8032a7476c099fb04877e5b6bce55d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_e8a91bff8f42093c941ca58c9512caf2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 320, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1663b2787ef0db1dde2e407c28723a82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8a91bff8f42093c941ca58c9512caf2
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ef3619de02b323bad3de8a88a2380e0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d10374948246dbdc8c79d52bcb42ba18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef3619de02b323bad3de8a88a2380e0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa4e7eecebba71d0c04eafd01c953090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(9216.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_42844b690326291e4cdd8ea95897c54d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_369a28e5bd1118559452fb793bdf0f8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42844b690326291e4cdd8ea95897c54d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b9f61dd232927c655bbcbb098b56d50c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 1, 960, 960], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5bfb975d2627f2446fea01ed0b5077a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9f61dd232927c655bbcbb098b56d50c
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_6f71a7634769cdc132273c797da6838d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4943d5957299860748a46c4b99879007
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8400, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f2c1de82a6a9883ab9bbd8cb93d93854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(12544.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_7c84f2d13b6bd2bd2f44408312c90b78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42db93a8980b46d64b33163cb610c827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c84f2d13b6bd2bd2f44408312c90b78
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.053489383310079575], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.013971466571092606], [0.28006845712661743], [-0.1266230344772339], [-0.043917324393987656], [-0.01946902647614479]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_5281c1b55faba0a1ab69daf3a8d10c67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c84f2d13b6bd2bd2f44408312c90b78
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0627698302268982], [0.05509001016616821], [0.37398090958595276], [0.5137683153152466], [0.17456120252609253]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.0767413005232811], [0.33515846729278564], [0.24735787510871887], [0.46985098719596863], [0.15509217977523804]], dtype='float32').reshape([5, 1]),
        ]


class PrimitiveOp_e7dfc93ce6288b0a6324e22fcb56a08e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f8dc0f39877fdb553b9ec07c4e12c53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7dfc93ce6288b0a6324e22fcb56a08e
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(32.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f8dc0f39877fdb553b9ec07c4e12c53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7dfc93ce6288b0a6324e22fcb56a08e
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(32.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_19cc3bc34b50e1060b12ab2cef7ff1e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2442.190673828125, dtype='float32').reshape([]),
            paddle.to_tensor(7300.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_145eebcc629f4bde4ef7218a6f34ae58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1825, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c20e11c1fb1aa73047d2b4cb9c6b409b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_145eebcc629f4bde4ef7218a6f34ae58
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c20e11c1fb1aa73047d2b4cb9c6b409b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_145eebcc629f4bde4ef7218a6f34ae58
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a97c9314fb456f54b05eabd664d6e152(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-3379.698486328125, dtype='float32').reshape([]),
            paddle.to_tensor([-0.3018754720687866], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d4ebcdf619d404505d70dae152ebaa8f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b6dce49ccf8d1d76b66cab3797320534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4ebcdf619d404505d70dae152ebaa8f
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_344a75530a8a8d4501a0d1676342fac4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(335.2174987792969, dtype='float32').reshape([]),
            paddle.to_tensor([-0.3018754720687866], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fee08ae21eeb4469894dbe88db6ed719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1156.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3546e8dc8a6842daa2ebd91dc0d08e39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_910e7e08fb820db8de0d850e1c479db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3546e8dc8a6842daa2ebd91dc0d08e39
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_910e7e08fb820db8de0d850e1c479db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3546e8dc8a6842daa2ebd91dc0d08e39
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_99603eb965a06cc637c6f16eebe6c271(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 512, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d72ac10bd2f917af9d83893dec134653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99603eb965a06cc637c6f16eebe6c271
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_09f9c9507da7244963925ab3fcd8d64f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 19, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cdb6da65f9c6fe3afd49c4109ab1737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f9c9507da7244963925ab3fcd8d64f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ec56cb2a6f06f470ec2eba064972bba2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 2304, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c3f56c70f0fcfaec601e67975bf2049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec56cb2a6f06f470ec2eba064972bba2
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8c3f56c70f0fcfaec601e67975bf2049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec56cb2a6f06f470ec2eba064972bba2
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_39277b25967a2db522cbf9baee8c7c84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21760, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f29219c4942441f0563d878b0aa49c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39277b25967a2db522cbf9baee8c7c84
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f29219c4942441f0563d878b0aa49c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39277b25967a2db522cbf9baee8c7c84
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dae5a2d74f6dfdf3c08830dab354a8c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(65536.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_6045089f2062055bbc32b22cffcac536(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00bae2f51fd1dcf8acd9da63cc47488f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6045089f2062055bbc32b22cffcac536
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_00bae2f51fd1dcf8acd9da63cc47488f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6045089f2062055bbc32b22cffcac536
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_929a59320a8ce308a60b65bc64b15fd3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 198, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87b7371ede55879b65ee925c56ccf056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_929a59320a8ce308a60b65bc64b15fd3
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3d856a6003b5cf89f06eeb0f0e321b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(361.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_de479ce475c4f8712ca3ce831b14cfd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(4125.45166015625, dtype='float32').reshape([]),
            paddle.to_tensor(12348.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_6829184c13406dfe691c48d9fb3f0dd0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3087, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb62d2418d48d1eeded64d0cd63b5275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6829184c13406dfe691c48d9fb3f0dd0
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb62d2418d48d1eeded64d0cd63b5275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6829184c13406dfe691c48d9fb3f0dd0
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c6b07caaa00ab7f3fed4506ef308723(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(21151.060546875, dtype='float32').reshape([]),
            paddle.to_tensor([-0.17289334535598755], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_71128c7165e11a23e00606cf93c27a16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a0f92516910d7785a7132592a9dbb65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71128c7165e11a23e00606cf93c27a16
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_581c4f0317d75a9ebc3c417d4a937db4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-693.3037109375, dtype='float32').reshape([]),
            paddle.to_tensor([-0.17289334535598755], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_65534a5001d55ed4b298546ec18dff16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b736e7b4b0a8658962e75658a289a63c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65534a5001d55ed4b298546ec18dff16
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(16384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c0ae64e72e5aa4e75259ac44863cd3ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61e7931d86ec467c5b0e4630a2fa5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0ae64e72e5aa4e75259ac44863cd3ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_81bf2f1cee6052f714b65115c017329d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_feaa8016dd66d455fa3594dd3e3fc1fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81bf2f1cee6052f714b65115c017329d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bcc50d1910fe1297f2cb93d3e8f34ba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_241011960a8cb233f88498b30179eeff
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bcc50d1910fe1297f2cb93d3e8f34ba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_241011960a8cb233f88498b30179eeff
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c9b57a86948d003bf58bde8b53b06f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61592fc5fc967acf1cfbd8b026ad8840
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_048890a093f8fae44ee60cdb2d2208ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_048890a093f8fae44ee60cdb2d2208ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4a10e8b2dafe4aa606884349437207ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(16384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b264268af78d24055b659edf41a403e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d82f2dd989a37d0afb4232d6fe15d9e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b264268af78d24055b659edf41a403e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.04345923662185669], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c8daed9b9727979c396ac071efc66db1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18f0916e02c2e9c0a93228706b5e3167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8daed9b9727979c396ac071efc66db1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2704.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_324a042b2e220af09c5477a86aeb91d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_324a042b2e220af09c5477a86aeb91d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f048a6f81b6a92fc4feab114cdb775b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4624.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9c091cbba25152ff2749cf59bff8fda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9c091cbba25152ff2749cf59bff8fda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_477836b2efc66ca1e0c3afd9e02f7ad3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70f97efcd7e28c556e1ca436a03d3769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_477836b2efc66ca1e0c3afd9e02f7ad3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.4293835163116455], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d6d673aa53487facfa88844d66a76af7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_684a45190610185e6bc1126212e49bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6d673aa53487facfa88844d66a76af7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.04147614538669586], [0.23358313739299774], [-0.1406402289867401], [0.18773312866687775], [-0.28871142864227295], [0.27953481674194336], [-0.060519345104694366], [-0.08555299043655396], [-0.13173845410346985]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_a40d66170d79c594bae09f022e732082(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6d673aa53487facfa88844d66a76af7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03344649076461792], [-0.23797376453876495], [0.19859591126441956], [0.15707193315029144], [0.33366891741752625], [-0.26910898089408875], [0.2096095085144043], [0.2242359220981598], [0.3340747356414795]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.07492263615131378], [-0.004390621092170477], [0.057955678552389145], [0.3448050618171692], [0.044957485049963], [0.01042583305388689], [0.14909015595912933], [0.13868293166160583], [0.20233629643917084]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_dcbaccedab1fbd8c70d4a7734e4d031f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 32768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 19, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97312a1d00bf04718ff4b9c9ff745912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcbaccedab1fbd8c70d4a7734e4d031f
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[20745.349609375], [20710.193359375], [20704.73046875], [20691.568359375], [20693.251953125], [20698.306640625], [20790.16796875], [20694.703125], [20689.958984375], [20618.73046875], [20700.275390625], [20644.333984375], [20679.34765625], [20690.94140625], [20714.51953125], [20695.16015625], [20702.21484375], [20714.3984375], [20740.71484375]]], dtype='float32').reshape([1, 19, 1]),
        ]


class TestPrimitiveOp_bfd3598d0cfa0438d6afb30cd7f346e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471753e5a1f43b1877e5fa9a2038ad6e
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bfd3598d0cfa0438d6afb30cd7f346e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471753e5a1f43b1877e5fa9a2038ad6e
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_db88c289ab42b30cb82c68bb2cb080d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8daed9b9727979c396ac071efc66db1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(3600.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_cb2c9b3aa1051c2cf862c2846a4f657b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac59e9a4b1c63ab87a02189c0330ba57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb2c9b3aa1051c2cf862c2846a4f657b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.40225356817245483], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e7d1c33a076036b9c76f1b2a034eff29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e7d1c33a076036b9c76f1b2a034eff29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_36859481ac5b0a1b864905ebaaee1bbe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_700586d765b3651572b907d33bc2eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36859481ac5b0a1b864905ebaaee1bbe
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_700586d765b3651572b907d33bc2eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36859481ac5b0a1b864905ebaaee1bbe
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d7259ed25c21573c6f2b68bc8aee9a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(6400.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2468c61eef5edcc5d7aedb9ef8aeab54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2468c61eef5edcc5d7aedb9ef8aeab54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_96c0285578a361ab3c8caf21e0aa9630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2793.063720703125, dtype='float32').reshape([]),
            paddle.to_tensor(8476.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c2570d6a870ff5892b7519b7845a20d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2119, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8efe88bf3b351f211b82c9362fbc075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2570d6a870ff5892b7519b7845a20d0
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e8efe88bf3b351f211b82c9362fbc075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2570d6a870ff5892b7519b7845a20d0
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aea6fb795bd6076254b1b6199aa5e3e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-15236.60546875, dtype='float32').reshape([]),
            paddle.to_tensor([-0.3848743438720703], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_55e9561d05e69f9fa539fc84a11d0141(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e53a6a9db68e44ac338ea18c822d1242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55e9561d05e69f9fa539fc84a11d0141
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c69b4a9e90da6777b992fc1eac90f509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-134.0641632080078, dtype='float32').reshape([]),
            paddle.to_tensor([-0.3848743438720703], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d10374948246dbdc8c79d52bcb42ba18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef3619de02b323bad3de8a88a2380e0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c9c4143af5c5597fcc1f13bbfeb25a29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_849ec7c645ad4a347bd586ebdbc99fcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6c726bb1b70d4e89232089767898c32f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(65536.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_7045764ff1606a1c2114ef7d98ab1c2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c086e20239c0eab988be062a81709fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7045764ff1606a1c2114ef7d98ab1c2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_18e226e95ec5d5c9402c294fc5bdc983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2704.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_9935249d185b359b70b3eaa5cb664cdf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88ad5af877d7ae7ac81818d02e68eafb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9935249d185b359b70b3eaa5cb664cdf
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_88ad5af877d7ae7ac81818d02e68eafb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9935249d185b359b70b3eaa5cb664cdf
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ec13f3817562484c7fc8cc9a39ff8bce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 1174, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_446b04b1837d2c5593def4aa5ccde2a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec13f3817562484c7fc8cc9a39ff8bce
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5520a993be93fa1c89fc6ffcb396f317(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4725, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b6aef4037934d46de62b2ce61b37a99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5520a993be93fa1c89fc6ffcb396f317
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4725, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0928d1a760bb9d25cd90186c07cd625e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_465b67050c3c534fe14d458e3bb0ce15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0928d1a760bb9d25cd90186c07cd625e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0dde66fdad7921245d4b44698087530a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-4.579158782958984, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2065899fc5ae213d249088920bfc29ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42d163917b0f74633d2d839e442f073e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2065899fc5ae213d249088920bfc29ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(676.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d6cc222d6b5f0acdf73a312098603926(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 11109, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c6bb989f053e65d1544847c5234da2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6cc222d6b5f0acdf73a312098603926
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 11109, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6cd0d6b6c47c653f54b0656d8b0e756d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b24a0fa94d38bbdc9b27551a5c7b0cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cd0d6b6c47c653f54b0656d8b0e756d
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b24a0fa94d38bbdc9b27551a5c7b0cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cd0d6b6c47c653f54b0656d8b0e756d
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f0d6cf9ea9c501de534a291dafa36f94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 198, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c128a8fc567fdcd8e10bdf82f09d949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0d6cf9ea9c501de534a291dafa36f94
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_de7abdafbdfbb78ac7fbbae0f9129709(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba01e28ec794de5339a9e52598ebd275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de7abdafbdfbb78ac7fbbae0f9129709
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4096.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a6fb96f392ea15a3c9a5434025b43e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25af0ca8b1099f1a176e9d4b52013b77
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(225.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_71bc27fdc515f7066a0529ee95b67e66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(676.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2ac5c549bcee32d1841a691ecc996542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f30c199bb51ae1811aa9569b2f82226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(0.04768742620944977, dtype='float32').reshape([]),
            paddle.to_tensor([3.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e01ae197afd5aa161f2954e101347f83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 9216, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fdc5d7feb41d1d34783831e2f42cb45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e01ae197afd5aa161f2954e101347f83
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4fdc5d7feb41d1d34783831e2f42cb45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e01ae197afd5aa161f2954e101347f83
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_43a8264387599a41d2d64444ae2a10fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2065899fc5ae213d249088920bfc29ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(900.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f81a275e11e5a83f053f0f79cd857a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(784.0, dtype='float32').reshape([]),
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


class TestPrimitiveOp_8294ecbef09416e8a627065175dc9159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16d8bd42f7eff78b9e3230066b7a7608
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12096, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b17c511cd99e1d7e78f1258107103a16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(576.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_94a67a6b88168f05f1fe8eb9f0eb46ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d090f9e82b68fabd1bb30894e3b535fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94a67a6b88168f05f1fe8eb9f0eb46ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6d32915e6cdc772607e041514b8f38a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32757673a5de7770d7814034a17a11d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d32915e6cdc772607e041514b8f38a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0671bca000ad30363b482cf27fed5a16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 8192], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8192, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b9bb3ffd940f6c74c8bea8fe184c748(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0671bca000ad30363b482cf27fed5a16
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8192, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1b5503add412187f8e75c884e9eb3420(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1d85d7291c6bfb87ea1bfe02a42554c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b5503add412187f8e75c884e9eb3420
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.12773509323596954]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_fe339a8302adfe9c2b78fc91bca17e92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b5503add412187f8e75c884e9eb3420
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2726812958717346]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.40041637420654297]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_4c93e703bb68b6e2ca766cb0487d6aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4c93e703bb68b6e2ca766cb0487d6aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_700586d765b3651572b907d33bc2eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36859481ac5b0a1b864905ebaaee1bbe
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_700586d765b3651572b907d33bc2eaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36859481ac5b0a1b864905ebaaee1bbe
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_7019d2e489022fc2063bc30d027b50b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 50, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02a501727237e9c8c22a0977711bdf88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7019d2e489022fc2063bc30d027b50b0
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 50, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a68c293f8857515be1b9709e6696eb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a621c5d797a61a537bd3d1833d32e41
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6069, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_32757673a5de7770d7814034a17a11d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d32915e6cdc772607e041514b8f38a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_df2d5832903d9027b874ccd5d4ce9918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(25600.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_046d00e99ce2fe931d0733ca5796b219(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 576, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e312d8e9e4602e57d44bc4329dc3161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_046d00e99ce2fe931d0733ca5796b219
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5e312d8e9e4602e57d44bc4329dc3161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_046d00e99ce2fe931d0733ca5796b219
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4d20f3a7d7b167039795995ba02b2d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6381ac9adbaa89dec0c7594136501af0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4d20f3a7d7b167039795995ba02b2d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6381ac9adbaa89dec0c7594136501af0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ee6de3d37a61c243380b5404215698a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1025, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_86f49df1872087bee30d1c01098f08fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee6de3d37a61c243380b5404215698a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b2c23d755f2224cc7ffc2d2609055ea3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba9981dd68ba200d455f6a39c254ed24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c23d755f2224cc7ffc2d2609055ea3
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ba9981dd68ba200d455f6a39c254ed24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c23d755f2224cc7ffc2d2609055ea3
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_519407c9823bdee6d6b86feab6d11c70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7bf8c384aaef90ca77b0c71455736047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519407c9823bdee6d6b86feab6d11c70
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7bf8c384aaef90ca77b0c71455736047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519407c9823bdee6d6b86feab6d11c70
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_39d555b3e5c8721d2d6273764738e2a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3db7e54fc98702a6a719d370673b455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d555b3e5c8721d2d6273764738e2a5
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c3db7e54fc98702a6a719d370673b455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d555b3e5c8721d2d6273764738e2a5
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_04febfe6e5f20f763ca0126890a31c04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 1, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d347c6abc84c6222fee5e77bafcb8927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04febfe6e5f20f763ca0126890a31c04
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d347c6abc84c6222fee5e77bafcb8927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04febfe6e5f20f763ca0126890a31c04
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cff609ee882a2a654110d543d9fd4b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ef74aa9f50b42fe3d36a1229797140
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_98df7e833d22dd8209df20d8344410e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2065899fc5ae213d249088920bfc29ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4096.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d62f6934c7efb373edc79b7c2f78c9af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65534a5001d55ed4b298546ec18dff16
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(7056.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_b3fe745b9daf6aedfac4e4ae754a0722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-0.7275834679603577, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_477f0876f75200ca93839878e37d3bce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_892e1ccd7414c6585e5ce259ae1abb49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c7a4be3a8fbfed1cba7577260102b48
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ea93b12b86017ec6b8b746792857ae4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(7524.00048828125, dtype='float32').reshape([]),
            paddle.to_tensor(22424.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_37305f20af6751259ca432e92ecda2e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5606, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f96c154327e97ffdc750d799cb4a52d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37305f20af6751259ca432e92ecda2e7
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f96c154327e97ffdc750d799cb4a52d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37305f20af6751259ca432e92ecda2e7
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3915c89d96ad30e6a0bd8faff87764c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-54164.71484375, dtype='float32').reshape([]),
            paddle.to_tensor([-0.445712149143219], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_31e286cf2491c5d2c6f4a68e8a707841(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5fbae43f49e0639017ef41d8c81a1e5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31e286cf2491c5d2c6f4a68e8a707841
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_63c44c9a106801b7d9d0a5e60214ea6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-46.57309341430664, dtype='float32').reshape([]),
            paddle.to_tensor([-0.445712149143219], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_418dd6ed100920388bc95f00852dcd3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8400, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc7adaa6c09a9e8d089037733b46d970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_418dd6ed100920388bc95f00852dcd3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8400, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_786f7525ab558f95884fa470895c2967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de7abdafbdfbb78ac7fbbae0f9129709
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1764.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a335037e6b6da03dd192d43505998719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76b5ec1e1c4784a6ec33688d260f4c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7434062c8ceb998dd2dc7cf6c572d719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5519ba142a258f164da420bf7b4c1d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1024.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d090f9e82b68fabd1bb30894e3b535fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94a67a6b88168f05f1fe8eb9f0eb46ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8e1327425ca0d53950538380d55d1398(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(1367.8251953125, dtype='float32').reshape([]),
            paddle.to_tensor(4144.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_99f4674c9a7106f00037e186ebcadb29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1036, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d568a65d8d3dfa2b08f871b2aecd926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99f4674c9a7106f00037e186ebcadb29
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1d568a65d8d3dfa2b08f871b2aecd926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99f4674c9a7106f00037e186ebcadb29
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0ad3df64a219777fc3a677f243f44c95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(2997.57470703125, dtype='float32').reshape([]),
            paddle.to_tensor([-0.05407264828681946], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d2155d51ea5f17ab7038280e743448b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4dbe7b82e55e78d771b9d4b4f958e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2155d51ea5f17ab7038280e743448b3
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_56768cfd6edbea744099f0544ea68a22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(75.52578735351562, dtype='float32').reshape([]),
            paddle.to_tensor([-0.05407264828681946], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e7d1c33a076036b9c76f1b2a034eff29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e7d1c33a076036b9c76f1b2a034eff29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_664e1bac33f6be372bec556cc48137ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(2384.88916015625, dtype='float32').reshape([]),
            paddle.to_tensor(7236.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_a9ffabd0253271741cb98e11e6fb2225(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1809, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_289d40195a414d2b879b0a93bed5bd2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9ffabd0253271741cb98e11e6fb2225
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_289d40195a414d2b879b0a93bed5bd2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9ffabd0253271741cb98e11e6fb2225
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_051e214aebb218acb3917e6bf5556286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(10918.9384765625, dtype='float32').reshape([]),
            paddle.to_tensor([-0.27381494641304016], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_975f16eb6b87fcb607411605655b3a2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3adb159b2b1e3f0eaff2bf6ad915f6a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_975f16eb6b87fcb607411605655b3a2d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f106c34527b5d0a7f9f49cb48c70d8b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(197.27517700195312, dtype='float32').reshape([]),
            paddle.to_tensor([-0.27381494641304016], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_231bf319a3acb2a59a477d9330779724(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf2d82e3c734412d6c7a771722a1cb49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_231bf319a3acb2a59a477d9330779724
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bf2d82e3c734412d6c7a771722a1cb49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_231bf319a3acb2a59a477d9330779724
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_f20a422e1c3323253e1ff0e283d03bc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 197, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25dd2d6e9124384fb8208ef07ef10eb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f20a422e1c3323253e1ff0e283d03bc7
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0f95f83b55bcb2932dc1bc1ca6ce6dbf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5bb847fd2c4f64ed450a5170af13ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f95f83b55bcb2932dc1bc1ca6ce6dbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 13, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_23f66bd6b8f32ce0dd944bb00acf034a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_903d771fb9b98c39b7ba6f54396deb92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23f66bd6b8f32ce0dd944bb00acf034a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3b1e932786102ead50e7a80392afaeb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(289.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_afae707c7993638537d26131333dde98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1f39f603d43ab4921d299a6df37b647
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6771823370743869100da560aecbd0cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eceeae894259106eef06544586420874
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(169.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_3df199e5e72b5d438cad6b0fcf58582e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 4096], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4096, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d68873c9ee567ffd8d65e8dc0b61225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3df199e5e72b5d438cad6b0fcf58582e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4096, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2ebc7db99b1d61dec15d7e951f04183f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aaaa64cbdaef9d69fa69ac5497e7945
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_25f47c97a4036b866c4becaa9afd2981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123458163592094d66722fa8aadcafd9
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_25f47c97a4036b866c4becaa9afd2981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123458163592094d66722fa8aadcafd9
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a69e06d82f440dfa469849299ae89096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78b9fbfda75fc362683e18e110f8a74c
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a69e06d82f440dfa469849299ae89096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78b9fbfda75fc362683e18e110f8a74c
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dd47e633ed68fe47dc56876f4358d2db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bed0a8c1232d259af5d750ac1058037
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dd47e633ed68fe47dc56876f4358d2db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bed0a8c1232d259af5d750ac1058037
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a8617ddb79b89aa74ff1dae2373aedbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_200e6629d14ef72ab61bc26745d6b3f2
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a8617ddb79b89aa74ff1dae2373aedbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_200e6629d14ef72ab61bc26745d6b3f2
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_2c1f6d046045d5904bba7e9c1c196f8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fdd1357fd8c9eccfb96a79a5559d144a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c1f6d046045d5904bba7e9c1c196f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_fdd1357fd8c9eccfb96a79a5559d144a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c1f6d046045d5904bba7e9c1c196f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_07fe5c32a9e045d6e17f00b555b92d04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 577], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 577, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_879377e6c2b88ea4f84eae3bd79ff3fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07fe5c32a9e045d6e17f00b555b92d04
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 577, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_126a1e12f026cd259994a3dce0ce33ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-3.758200168609619, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_53f4bf1462db00f34934a82df59137a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a94ce4fcadd171d195584afa2bfe7d
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bf4c9ff9eb35790557ff2378ce0d8749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb1495e94fb16d27855af6b6599266b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_05d21a58ad0b272031b0bd2636ffef17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(3136.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_d082bdf7603af82dab82cb6c6cd77191(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1412b98f9867a93262fc89ac68140082(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d082bdf7603af82dab82cb6c6cd77191
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 25, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_261bd3021b91c14faf27cd76942437f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 1, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90f54464743169aedc62f511a7cf0448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_261bd3021b91c14faf27cd76942437f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4fdc5d7feb41d1d34783831e2f42cb45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e01ae197afd5aa161f2954e101347f83
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4fdc5d7feb41d1d34783831e2f42cb45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e01ae197afd5aa161f2954e101347f83
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ba9981dd68ba200d455f6a39c254ed24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c23d755f2224cc7ffc2d2609055ea3
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_ba9981dd68ba200d455f6a39c254ed24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c23d755f2224cc7ffc2d2609055ea3
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(64.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7bf8c384aaef90ca77b0c71455736047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519407c9823bdee6d6b86feab6d11c70
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7bf8c384aaef90ca77b0c71455736047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519407c9823bdee6d6b86feab6d11c70
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c3db7e54fc98702a6a719d370673b455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d555b3e5c8721d2d6273764738e2a5
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c3db7e54fc98702a6a719d370673b455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d555b3e5c8721d2d6273764738e2a5
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d347c6abc84c6222fee5e77bafcb8927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04febfe6e5f20f763ca0126890a31c04
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d347c6abc84c6222fee5e77bafcb8927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04febfe6e5f20f763ca0126890a31c04
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2468c61eef5edcc5d7aedb9ef8aeab54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2468c61eef5edcc5d7aedb9ef8aeab54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9aa7ecb13b166316c2c5c04c93db62b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9cccd621cc3421138f9edabf34494595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6788cd52d4620f93775337d648c1c774
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9cccd621cc3421138f9edabf34494595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6788cd52d4620f93775337d648c1c774
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(256.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_51de3ec31ab1ea3848e4c265af096298(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_179aa4a0bf7814176c0ed811af7bb076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51de3ec31ab1ea3848e4c265af096298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.15813791751861572], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d987f8ca2b7f7708dd5581551b7e5071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(10816.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_b96bf5cb1822fdef70540ddc601a781a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 152, 272], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b5d022a6be5d470cfb00546d630e42f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b96bf5cb1822fdef70540ddc601a781a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d3d86648b8c5fef0c642aa3efce63e80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(10816.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_df8207840e16cfbb4239d49254255314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad6ecad809d350691b47c24d8a26d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1335b58afd0bccf325beb2e500e46bf1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 144, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_875cba435f696ba853df0f791fd88a1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1335b58afd0bccf325beb2e500e46bf1
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_875cba435f696ba853df0f791fd88a1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1335b58afd0bccf325beb2e500e46bf1
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9f3310466b56fc07e61f7e9089e0f6e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(33856.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6edacad87532b60b2952a4de7a5a5e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(9.757837295532227, dtype='float32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2751b5588a28ec4d0ede7b60cac307d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(0.38435766100883484, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_61e7931d86ec467c5b0e4630a2fa5050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0ae64e72e5aa4e75259ac44863cd3ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8032a7476c099fb04877e5b6bce55d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8032a7476c099fb04877e5b6bce55d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2794f233f52bd7c24e3e81875da463e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2794f233f52bd7c24e3e81875da463e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_593358987e9421e08775fa838144a35c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(5776.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_cdfff13bd38db149b8bec0edba751f69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9288920304028b1ae986841ae437f48b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c536d82721744cc0af3ecbee4d7235a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30f14c8d377f60e0c7bb8a31a199fcc
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d3c8fbe01c894cee527d24e35178980c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(5622.19140625, dtype='float32').reshape([]),
            paddle.to_tensor(16716.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c1102010e7694e93840fd62fb1a6afe4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4179, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_424d1f83a3745c5697278093c0298312(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1102010e7694e93840fd62fb1a6afe4
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_424d1f83a3745c5697278093c0298312(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1102010e7694e93840fd62fb1a6afe4
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1ecbd5f8611d500b958820760007f70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(694140.625, dtype='float32').reshape([]),
            paddle.to_tensor([-0.013174593448638916], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6cb87cf22218f2f65edc7b796bf4cf2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e09fd630bcb2646e12fe37a6b6ddc352(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cb87cf22218f2f65edc7b796bf4cf2f
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_a809624e9888a4e87b1d1ccb744fdf27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-48.72361373901367, dtype='float32').reshape([]),
            paddle.to_tensor([-0.013174593448638916], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e390f650bcdbadca4070bdaaab94f216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f209460593495992a15bc5caa65ec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(65536.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_029cdfebc5a2e02580b06d90a235676f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a47274817015a7349f92259db4706444
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c086e20239c0eab988be062a81709fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7045764ff1606a1c2114ef7d98ab1c2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_27345eb8c1a6e317b2bcdccdcde70b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4096.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5866e07ffc06e786e8cc63a7a8559fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dffe5ee239f91530cab66dd52f24ef43
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6a1ae9e01e3b770b47b7234da96ee18c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(0.22050422430038452, dtype='float32').reshape([]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_4f49de5cc14ef58fbd9368ca2b046dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e1f1271690e70375c15c3d7e621d6eb
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6804, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a6564283e7cbf797afc052bb56be0776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_345d2d085813649aee260a0dfb3ff397
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5376, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e41c5f5b0e09e6ceeace408d23f6dc58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8daed9b9727979c396ac071efc66db1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(16384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e1e9dccb09d1c802cba2d905c80aa8e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e1e9dccb09d1c802cba2d905c80aa8e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_6332714bfb39069f6630d2611dbca3e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(8464.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8032a7476c099fb04877e5b6bce55d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_8032a7476c099fb04877e5b6bce55d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ac828c055996baae99cc2e1e97ae67
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1663b2787ef0db1dde2e407c28723a82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8a91bff8f42093c941ca58c9512caf2
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_55cf3e66d20d0a852e4bfd79fb1c3909(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 144, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57eaa5a45d7062a71d56c1a8a5e14604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55cf3e66d20d0a852e4bfd79fb1c3909
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_57eaa5a45d7062a71d56c1a8a5e14604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55cf3e66d20d0a852e4bfd79fb1c3909
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(768.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_1700a303d535bba98fa8f11e74c7d9d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(400.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7671e26b8bd3d491857ae98c218a04f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-3.6430866718292236, dtype='float32').reshape([]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_10cc8e28f419efe54e333e8cac987b3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(6234.8984375, dtype='float32').reshape([]),
            paddle.to_tensor(18648.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1b46adfc873cf06869082f786a2ac937(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4662, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13970dcec7b8348a85cf17a5585be10b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b46adfc873cf06869082f786a2ac937
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13970dcec7b8348a85cf17a5585be10b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b46adfc873cf06869082f786a2ac937
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_16e6d661b75bf9c780ecd6258659c6dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-57442.97265625, dtype='float32').reshape([]),
            paddle.to_tensor([-0.2728491425514221], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_90d444bbfac5fd5c0e7e745a63566449(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50c823107566590f2055297489e10ba4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90d444bbfac5fd5c0e7e745a63566449
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_7b171daf8746fccee9439f8fdd2fb41c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(364.6025390625, dtype='float32').reshape([]),
            paddle.to_tensor([-0.2728491425514221], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_790b164c6511ad385f00bb714ace8273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3546e8dc8a6842daa2ebd91dc0d08e39
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_790b164c6511ad385f00bb714ace8273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3546e8dc8a6842daa2ebd91dc0d08e39
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(512.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d72ac10bd2f917af9d83893dec134653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99603eb965a06cc637c6f16eebe6c271
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_048890a093f8fae44ee60cdb2d2208ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_048890a093f8fae44ee60cdb2d2208ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e714cc16ce2ccdf5566e9d0436e6f3de
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(192.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e1e9dccb09d1c802cba2d905c80aa8e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_e1e9dccb09d1c802cba2d905c80aa8e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772ec8c703c44f4b4ed17fa76de85383
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d0a488db8813c5724edfe9110d1a935c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04c4764624c6fb3b6b986eeb32e5ada8
    def get_inputs(self):
        return [
            paddle.to_tensor(5113.6044921875, dtype='float32').reshape([]),
            paddle.to_tensor(15428.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_0f42919fe6cf6ebd6d56b4f8dd809060(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3857, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82fbec6a74885b3626b268d5f2184165(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f42919fe6cf6ebd6d56b4f8dd809060
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_82fbec6a74885b3626b268d5f2184165(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f42919fe6cf6ebd6d56b4f8dd809060
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29d4177beb630c5555ea53330702486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(83241.71875, dtype='float32').reshape([]),
            paddle.to_tensor([-0.3943657875061035], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_01ea2eb299548546500eaf320dc2dcad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4bc6cb24bb0763ad3cc4d9824e77ba15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01ea2eb299548546500eaf320dc2dcad
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_dfdd38dfb22d9a9384d5c914742e6e26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9b6439532c9ba79072152eb89e60f87
    def get_inputs(self):
        return [
            paddle.to_tensor(-69.41327667236328, dtype='float32').reshape([]),
            paddle.to_tensor([-0.3943657875061035], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9c091cbba25152ff2749cf59bff8fda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9c091cbba25152ff2749cf59bff8fda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2248fee1ba62d0098c3f4d5b87e40043
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(128.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_799fd4483319a2fa7e336cbaf1de6ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f05a1746a091fe779f485a29168fe33
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9d9cda82bcd589bfe3aac88b8b4590b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9935249d185b359b70b3eaa5cb664cdf
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_9d9cda82bcd589bfe3aac88b8b4590b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9935249d185b359b70b3eaa5cb664cdf
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_27ad0336033287ca716c19f2b8765ccb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1174, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d69770efd2b864912762fac7cf34a9be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27ad0336033287ca716c19f2b8765ccb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_465b67050c3c534fe14d458e3bb0ce15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0928d1a760bb9d25cd90186c07cd625e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_965b5ece88626f770040c7586420116d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6069, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f37b954fb28357c1af1ff1731830de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_965b5ece88626f770040c7586420116d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6069, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a00cd5201d5e3f222b210b8c7ce74030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(2304.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_5b496af442bd83898aeb4b722df7bb82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6edb276fa31ded0dcef7a123d665c8f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b496af442bd83898aeb4b722df7bb82
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ee41dd55e54ae35649780b5c5e0ca0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e13cdb09e433b59dd353aef918254000
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(1444.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_957e930e2de436bedb2337ad5900a802(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3024, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57965f78aadb7ce517a5d4bf7bfca437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_957e930e2de436bedb2337ad5900a802
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3024, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_728e433aa5805bfb4f5d20ce41bc2c8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c00becb5d5847f33a98dcc8ffb177c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(100.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5c2a839593a58355c519efb1891fb92f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_555260b46f5d4a6a72a025806b8cdef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(23104.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_c94e349a450aa3b93418d1d332cdd7ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 576, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a4a0105bed41b91d7ef16dc88832c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94e349a450aa3b93418d1d332cdd7ab
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2a4a0105bed41b91d7ef16dc88832c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94e349a450aa3b93418d1d332cdd7ab
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(384.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_369a28e5bd1118559452fb793bdf0f8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42844b690326291e4cdd8ea95897c54d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d969c58d698541fab4258ecbb749def2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eceeae894259106eef06544586420874
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(196.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0fa07117f0d3e7784b7bd3e3a8506ebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9765427904702b3cec29af184420b13e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(16384.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_1b55e6c640098b1b5f55c96425d2b8ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 38, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_608dd93ceefdfed724b7e8e027de3c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b55e6c640098b1b5f55c96425d2b8ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_371f29e24c5bcdee7bf4256a38c49f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcff2a22176319290947c71a4a6ef2fc
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(96.0, dtype='float32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()