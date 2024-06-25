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



class PrimitiveOp_fee06b3c62f9c2d2b9c6410486e602f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0, 1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cb11438680ca91d8702808779b41ee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fee06b3c62f9c2d2b9c6410486e602f7
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6cb11438680ca91d8702808779b41ee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fee06b3c62f9c2d2b9c6410486e602f7
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_64de2bcc4923cef417864b3c81aed40b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89edf236b79e334bef5a2cebbe4f1478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64de2bcc4923cef417864b3c81aed40b
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b382e26a9bf27a1fe56df529dacfaaa7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b0f14fa3eb447894e9b30fa9739c676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b382e26a9bf27a1fe56df529dacfaaa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_b69f2f04b3edd412c98cc3320c15ee38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_204b39d9ba88b6cc6c921258b664b99a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b69f2f04b3edd412c98cc3320c15ee38
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_68be8ef97967a99444ee7d9c50c5dad8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a19476b26f1e2db60d8bc8790e7f9627(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68be8ef97967a99444ee7d9c50c5dad8
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_f69a9b0a4f10d10722dd038ff3530035(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60f90ac3970751375abe58f7054401c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f69a9b0a4f10d10722dd038ff3530035
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_6f7b5d873258f21e1ed2b73aba203065(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf667beb493f3a1e6ec86bd0b7e00bf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f7b5d873258f21e1ed2b73aba203065
    def get_inputs(self):
        return [
            paddle.uniform([4, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_c9dd82d6f89f7b29cdb92c3d8a48a2cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c831c6fab89499809bc35ee98fdcfe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9dd82d6f89f7b29cdb92c3d8a48a2cb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4868095815181732, 0.33942610025405884, 0.4916043281555176, 0.0018385753501206636], dtype='float32').reshape([4]),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a7b6ab04f76328231114fc3301452de9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07e6d04a48b25d3e141ed176424c4ba9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b6ab04f76328231114fc3301452de9
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_35c9d9bad84da7e490093f855fdf349c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d83c416220e25bd04ba51968a8228bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35c9d9bad84da7e490093f855fdf349c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9ae0b024347c42d115b1e0762b868717(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 500], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7adf9e1645f0b1751954ea5b989121d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ae0b024347c42d115b1e0762b868717
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_57094ea9c93330fe82f1162ee7f22407(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 500], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_84a315f151cc05d6c697931fee6a0766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57094ea9c93330fe82f1162ee7f22407
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9f90491493467f9408724e5163b307ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad04d7d29e33e3dea97b58deac86cccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f90491493467f9408724e5163b307ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_3d5af038d7dd8581b2890036111d0b52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8732], dtype='bool'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9c821021a124c4b6d2c4d953fb0b48e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5af038d7dd8581b2890036111d0b52
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_995b3bd61124ba64ffcdd7e98567fa4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e18e0469c799484e8fd3d7b248ad483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_995b3bd61124ba64ffcdd7e98567fa4d
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_39141faa212684fd66200dfa870874e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[21, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91bd3838e1059611f6de058c2722f1b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39141faa212684fd66200dfa870874e8
    def get_inputs(self):
        return [
            paddle.uniform([21, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7adf9e1645f0b1751954ea5b989121d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ae0b024347c42d115b1e0762b868717
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84a315f151cc05d6c697931fee6a0766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57094ea9c93330fe82f1162ee7f22407
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5e18e0469c799484e8fd3d7b248ad483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_995b3bd61124ba64ffcdd7e98567fa4d
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_81447b832589652a97899a10ef12b771(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21824, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3843e76c6574c0c13c0e3af90979f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81447b832589652a97899a10ef12b771
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_60f97ce894708a5dfdf30b55a5f933ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_737240d560e9b4eaba581a0e20a30fbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60f97ce894708a5dfdf30b55a5f933ff
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.46265488862991333, 0.016697153449058533], [0.43998315930366516, 0.12417177855968475], [0.2822396457195282, 0.03765062242746353], [0.15949825942516327, 0.06882414221763611], [0.10706356912851334, 0.3038007915019989], [0.2781323492527008, 0.09815538674592972]]], dtype='float32').reshape([1, 6, 2]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d4e6578ceb2600eb4a49c9afd5ab5adb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_081f7450fce8978b7d6ac26e5f6da2aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4e6578ceb2600eb4a49c9afd5ab5adb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_db8c277c7246ef50b52e4c9e7da9c923(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_05cba2b6a4faf6a028edad8375b6d4e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db8c277c7246ef50b52e4c9e7da9c923
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a8c3f8f66d0d5ea3970dead6f011375d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db8c277c7246ef50b52e4c9e7da9c923
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4ca460b5eeace799e540078047be11f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1709cadbcc10fa9282f9f200ec5032b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ca460b5eeace799e540078047be11f4
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_05484f78e9b87a1ad7f8a6a3517d8a5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ebe9f1f9197e50531af0bcb48fb537a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05484f78e9b87a1ad7f8a6a3517d8a5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_91395a1c1ce040995e32bc5c80de74f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e13a2dc78ad02a080e9903ddc8521a4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91395a1c1ce040995e32bc5c80de74f3
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9b4af6d15c9bc95ccafd7aa673862ac3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8a2ddf5ee72eb6ebd2b46500077c693(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b4af6d15c9bc95ccafd7aa673862ac3
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.009557947516441345, 0.1171107217669487, 0.09393314272165298, 0.41896283626556396]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8d82a6c78f10eb16e64cd08fa528d436(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d5ec88a0850371337850a8da23dfb23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d82a6c78f10eb16e64cd08fa528d436
    def get_inputs(self):
        return [
            paddle.uniform([300, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_09bc2bbc58fb9a7508929589b8e33dc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 36, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03195b8b67b3b54c3a67870ce94172b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09bc2bbc58fb9a7508929589b8e33dc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03195b8b67b3b54c3a67870ce94172b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09bc2bbc58fb9a7508929589b8e33dc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_497d034e6bcb80081d0f2007898d7216(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c32a08e616e2a20d046fea26aeb009f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_497d034e6bcb80081d0f2007898d7216
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_52d3f951853c05baa967f39a33cfb395(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abc464f6c9e2379f01f43554a57ec8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52d3f951853c05baa967f39a33cfb395
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b8ced4c5071f04ac24673e967f73a50e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0, 1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a02286ea7dbe516dd5a754ec07415f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8ced4c5071f04ac24673e967f73a50e
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a02286ea7dbe516dd5a754ec07415f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8ced4c5071f04ac24673e967f73a50e
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_359af65b59d88e11fc8f5f1b48f97c0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 21], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_048269945db1be36e64643abc0b8623a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_359af65b59d88e11fc8f5f1b48f97c0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_028093063c15da65593fed9bb8c54d79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35ebee60641360c7424da3de52b00702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_028093063c15da65593fed9bb8c54d79
    def get_inputs(self):
        return [
            paddle.uniform([3, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1697caf8d62d1607a022f41bd06a47c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20fb40293ecd88657077ae69a75d3946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1697caf8d62d1607a022f41bd06a47c1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35455819964408875, 0.25965553522109985, 0.29125335812568665], dtype='float32').reshape([3]),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_07258fe36d82088c86974e617da22a33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d05e160bd045260e1682ce7e780e00bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07258fe36d82088c86974e617da22a33
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_60545acb49c533412285252116d674c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 872], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1187fb503b25aa311ecbbd3b22cae9c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60545acb49c533412285252116d674c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_58108b1a8be056b3b776c958652cacf0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66f9a457079f274d377a6f2dc717defb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58108b1a8be056b3b776c958652cacf0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a40db378e3fffdf81f3868130ee426b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_758cbfc6f4fbca58cce80367d74a43f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a40db378e3fffdf81f3868130ee426b2
    def get_inputs(self):
        return [
            paddle.uniform([1787], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e5dd1ccef59da0a608e6c43553753a3a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549], dtype='bool'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0196402e4ea60b452b59f352fb73f528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5dd1ccef59da0a608e6c43553753a3a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_59871990242fe4bf494564b5cabe977f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e1b0e497229832cdc229e1746c9e7a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59871990242fe4bf494564b5cabe977f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1787, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e1b0e497229832cdc229e1746c9e7a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59871990242fe4bf494564b5cabe977f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1787, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d5babd8fba03bda8a4beeb91f2979e2e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0, 1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76dd3f8d05e9295a211e366c67a5ead4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5babd8fba03bda8a4beeb91f2979e2e
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_76dd3f8d05e9295a211e366c67a5ead4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5babd8fba03bda8a4beeb91f2979e2e
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_f057a54a8a3e37b7b8d753953bd9d47b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 4, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c9b2a88ca981f8b55a6d29925c9d90c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f057a54a8a3e37b7b8d753953bd9d47b
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d8ef22e86cf2422acc6bb7ad2575d1f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b95da9e1fce4acd9ec79391355e6558(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8ef22e86cf2422acc6bb7ad2575d1f1
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_2431abc404a09433093499a8ad9f50d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 768, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c18a1437afe4571bb988ce984e6f3ba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2431abc404a09433093499a8ad9f50d6
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_762d0b6c7ee4d04148d56b54f296771a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fdfbce2b6c84d3cb80ebd9c232cc602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_762d0b6c7ee4d04148d56b54f296771a
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_39f8bd89b101347ba414881710563a60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ed343b7f5dfc9adad2a99867e827743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39f8bd89b101347ba414881710563a60
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5577a20a08134045abe09ef28eac9b97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4772d867b5080a95f0c3cfc04d518da2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5577a20a08134045abe09ef28eac9b97
    def get_inputs(self):
        return [
            paddle.uniform([5585], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_fe3c6954e5f790104dc262b9ecd66f4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109], dtype='bool'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d544e434615348829fea2debbdf5509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3c6954e5f790104dc262b9ecd66f4d
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bfba88bd19a651300a08153c755b486b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9520419e9a94b59869697c17efafa72d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfba88bd19a651300a08153c755b486b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5585, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9520419e9a94b59869697c17efafa72d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfba88bd19a651300a08153c755b486b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5585, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_65888d786e0157e67abffaef45c3673c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2275021629306111ffe5af52c4d2b8f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65888d786e0157e67abffaef45c3673c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2275021629306111ffe5af52c4d2b8f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65888d786e0157e67abffaef45c3673c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_60f90ac3970751375abe58f7054401c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f69a9b0a4f10d10722dd038ff3530035
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_468324d038ef45fe9a7bd877c01e86c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c86fdbd166a3260d4db00be9b5c2a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_468324d038ef45fe9a7bd877c01e86c4
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c2414590f88dfbe3fa4877e72272c914(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[19, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_316703c7af67632dba8315c56eb8b453(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2414590f88dfbe3fa4877e72272c914
    def get_inputs(self):
        return [
            paddle.uniform([19, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9963fbea0b5829b243b70311dee441f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cdd39c190fbc9f55067738fb5df9d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9963fbea0b5829b243b70311dee441f7
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ce21811858af63334c2db8cfd166ee88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d89c9815a1429af7bf8c0c20821ad38e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce21811858af63334c2db8cfd166ee88
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_66f9a457079f274d377a6f2dc717defb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58108b1a8be056b3b776c958652cacf0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f3bdb68276b69f3a20d24976a489b3f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a356806aa7ddb4a0c27425218218df76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3bdb68276b69f3a20d24976a489b3f3
    def get_inputs(self):
        return [
            paddle.uniform([1774], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0196402e4ea60b452b59f352fb73f528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5dd1ccef59da0a608e6c43553753a3a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d2e61a4e707f1f67619f10c9671bca55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b2b4b707464062235f81fc1bc2c655d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2e61a4e707f1f67619f10c9671bca55
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1774, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0b2b4b707464062235f81fc1bc2c655d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2e61a4e707f1f67619f10c9671bca55
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1774, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_95cdd94599e8c5f0d2924f4b320148ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 336], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29deccc009bddc79be589b26892f1d83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95cdd94599e8c5f0d2924f4b320148ee
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_48601144992a80ff71d27c825e5e8f71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_363d11f3d692ff4c6ed2cb1d55f95d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48601144992a80ff71d27c825e5e8f71
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1b95da9e1fce4acd9ec79391355e6558(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8ef22e86cf2422acc6bb7ad2575d1f1
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_5aa99892aae5fa4f0a272098f5aca47f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_646fac0c6ce4134b408c64dbc20edb71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aa99892aae5fa4f0a272098f5aca47f
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_83a21327893a9c1b30cdc037e76c1c61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d71757a3c54aa1abd30020f47fe5a36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83a21327893a9c1b30cdc037e76c1c61
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_62e2ddaca06e957fc26220380782a687(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_024be74abb2a46e4eebc0e3d0f7994bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62e2ddaca06e957fc26220380782a687
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_cd25b467ea7d9e3323ddaa4d4b9392c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15eaea8e2948f5745ddff02796fb6fad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd25b467ea7d9e3323ddaa4d4b9392c6
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6d5a10578ef7acd4086bca1b92b7afda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4652653772c31ee796747cf91439143e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d5a10578ef7acd4086bca1b92b7afda
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5a0ac08e9dd26e0ed32b0bd0dcdf46b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d5a10578ef7acd4086bca1b92b7afda
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9567fe0dfa0960e21b7fc12b6262da3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4765ad7998453a3c81932fccfd0b51af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9567fe0dfa0960e21b7fc12b6262da3f
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_c64217152f6d9c70a8d3be36822cacbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e140b07f154cef7c290ae5e678eb6ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c64217152f6d9c70a8d3be36822cacbc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d64aea342b60f32753119e8f4f97fcd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32b1e9dea5abaabceb0ad3253f5fd1fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d64aea342b60f32753119e8f4f97fcd7
    def get_inputs(self):
        return [
            paddle.uniform([1501], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_8518f8c217e1eae2e2ddfa983ad3a67b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024], dtype='bool'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be16bf3c012f68cf498c498ed054c7f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8518f8c217e1eae2e2ddfa983ad3a67b
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2ccc944ba348cbac980c4a8280c9cda6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc82372d18ed808ac3eec3a8711ee629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ccc944ba348cbac980c4a8280c9cda6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1501, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dc82372d18ed808ac3eec3a8711ee629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ccc944ba348cbac980c4a8280c9cda6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1501, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1e27c5177a4bbde20848134d0078f211(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fe53d67e539328f07b76749096837fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e27c5177a4bbde20848134d0078f211
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_25056672ab89efb55bd85f57acc72cea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 16, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b612c2c11926d7b697930769c428459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25056672ab89efb55bd85f57acc72cea
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9cece9461ad6c4fc7b3f049791f61a6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e974acf76db25be5350d7fd25892462(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9cece9461ad6c4fc7b3f049791f61a6c
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1ad0636438df8df0cb9f3c85ed0aebf2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac24e72ac5a00a9c306876f032286e1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ad0636438df8df0cb9f3c85ed0aebf2
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ec54f48f5bb7e36243f367d74f6bbf1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ad0636438df8df0cb9f3c85ed0aebf2
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c32a08e616e2a20d046fea26aeb009f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_497d034e6bcb80081d0f2007898d7216
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_abc464f6c9e2379f01f43554a57ec8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52d3f951853c05baa967f39a33cfb395
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_97dccfe0a3ac9d1a51c7189603875ee2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e4a8796bfc83983001945cc42ef4fab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97dccfe0a3ac9d1a51c7189603875ee2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bdb3ce3710eb3bce3b337b1e8899025c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_552c9f7649d319e430201b8e18b85c67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdb3ce3710eb3bce3b337b1e8899025c
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5b612c2c11926d7b697930769c428459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25056672ab89efb55bd85f57acc72cea
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c32a08e616e2a20d046fea26aeb009f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_497d034e6bcb80081d0f2007898d7216
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_abc464f6c9e2379f01f43554a57ec8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52d3f951853c05baa967f39a33cfb395
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b282e799aa2df19ffe20d35303613596(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91d624809a2495619fa53f14850f8f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b282e799aa2df19ffe20d35303613596
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_3ed5cb62a0da3a480d0699351fd16e77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc65d4ebf23b96e3b6e82ff2d064822c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ed5cb62a0da3a480d0699351fd16e77
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fc65d4ebf23b96e3b6e82ff2d064822c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ed5cb62a0da3a480d0699351fd16e77
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_9a61ef7c1fa5d4f90d4658a7ac403e6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7268e225bd868c064ddf9216d0b86791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a61ef7c1fa5d4f90d4658a7ac403e6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_2e69e8b7419a1611aa1cc362aa2fc3ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9340186050da1439c49accfb7c77dd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e69e8b7419a1611aa1cc362aa2fc3ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_29deccc009bddc79be589b26892f1d83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95cdd94599e8c5f0d2924f4b320148ee
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_15eaea8e2948f5745ddff02796fb6fad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd25b467ea7d9e3323ddaa4d4b9392c6
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1c16b8ff6fe14839d82f66dd890ab46a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16105083f5955bb4f70c5b160678f050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c16b8ff6fe14839d82f66dd890ab46a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_52ec454dc2b49a4ee78dddfb3799b2c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_199aafc6feaa303af3d353caa53dcbad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52ec454dc2b49a4ee78dddfb3799b2c1
    def get_inputs(self):
        return [
            paddle.uniform([2049], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_572bd22ae5f076d02d61bd33e4fcd0f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116], dtype='bool'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abc31e9cac760e11f68619b77f8f59d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_572bd22ae5f076d02d61bd33e4fcd0f6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_87cfd55708d8a495449cd6cf8c3250e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60fb5d2e1db323475cbc46f687c038d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87cfd55708d8a495449cd6cf8c3250e0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2049, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_60fb5d2e1db323475cbc46f687c038d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87cfd55708d8a495449cd6cf8c3250e0
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2049, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_29a532221c521a8008a883c9c53ed896(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5c0ac8626eb33b2a926f1f03e093121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a532221c521a8008a883c9c53ed896
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3cfa58e47d0ed4641658b97689175a0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b50d826e89a4c61c713a183ec7b8aedf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cfa58e47d0ed4641658b97689175a0c
    def get_inputs(self):
        return [
            paddle.uniform([4634], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c58c325872058c0fd3e1d3662eabd528(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261], dtype='bool'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ac5cdd08899b47a39048762e335e31a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c58c325872058c0fd3e1d3662eabd528
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0458cfa7d7c49ef51916cf4e51ad42da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d2249eacff35dc8497a3d38cceb9283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0458cfa7d7c49ef51916cf4e51ad42da
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4634, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4d2249eacff35dc8497a3d38cceb9283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0458cfa7d7c49ef51916cf4e51ad42da
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4634, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_96515aeef234eb0a001efb41923890ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1cf2cb40406ad697711665223c104601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96515aeef234eb0a001efb41923890ee
    def get_inputs(self):
        return [
            paddle.uniform([6, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_295b72a0872b3137cc0ea39404b49c59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_353250750bc74d3675dc28615821d667(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_295b72a0872b3137cc0ea39404b49c59
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2540411651134491, 0.26115354895591736, 0.014611984603106976, 0.3021100163459778, 0.47785574197769165, 0.49382632970809937], dtype='float32').reshape([6]),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ba4a73ea75e856e60bac47b9c772486e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2434], dtype='bool'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c6e7e103afa13a0528a8eba01e16f43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba4a73ea75e856e60bac47b9c772486e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_11323776b43e773883a3850de26c0e1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f5684feb556e572ff93e381380b2c3da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11323776b43e773883a3850de26c0e1c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_99ac083f728ba4c81f8fcdd93b6f844c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af1e31433cfa9d13e5d9468fcdb6466f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99ac083f728ba4c81f8fcdd93b6f844c
    def get_inputs(self):
        return [
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_56d5370bc3ef96ee786366b7da4cdf33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100], dtype='bool'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa5fc46b45b8129f6ad80329c95b951b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56d5370bc3ef96ee786366b7da4cdf33
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0078bdffeb979e4442eddbd96706e02a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b89f49dc7cfef7c8e5ebe3da4afedb89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0078bdffeb979e4442eddbd96706e02a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1000, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b89f49dc7cfef7c8e5ebe3da4afedb89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0078bdffeb979e4442eddbd96706e02a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1000, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_712343170c95890c8963221efd8ee201(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf428d8310734e6fa888dd7253484368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_712343170c95890c8963221efd8ee201
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2984451651573181, 0.050633374601602554, 0.2856460213661194, 0.48093360662460327], dtype='float32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_141ed1b4f796b032676402b7d486a86d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90d73213a870e9d871a3648b7182f273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_141ed1b4f796b032676402b7d486a86d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2984451651573181, 0.050633374601602554, 0.2856460213661194, 0.48093360662460327]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5191b06786e80e26c87f370cc6cfaded(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9784309cbf7309d278f01b192de450cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5191b06786e80e26c87f370cc6cfaded
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7e5b2f0e738413cbb09fc4609d2427e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 768, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a303ebad48585433b36736a5f85f66a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e5b2f0e738413cbb09fc4609d2427e6
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_024c4f0c07e3ee74425f812dbbddc555(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22a1e3568ec7e399fff742394f615dc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_024c4f0c07e3ee74425f812dbbddc555
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c32a08e616e2a20d046fea26aeb009f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_497d034e6bcb80081d0f2007898d7216
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_abc464f6c9e2379f01f43554a57ec8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52d3f951853c05baa967f39a33cfb395
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_21f0d30ecb51b668193b9ce20c33eb71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_712343170c95890c8963221efd8ee201
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2080528885126114, 0.13153503835201263, 0.44382715225219727, 0.4816720485687256], dtype='float32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9b38469b237a2235cc468458a6852e44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_141ed1b4f796b032676402b7d486a86d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2080528885126114, 0.13153503835201263, 0.44382715225219727, 0.4816720485687256]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_89b54a6d74fdcaa1faeb8f0dc966dd8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34d2006409d3c400c2e5d2b0e5bb8762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89b54a6d74fdcaa1faeb8f0dc966dd8d
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1f8d25c85a44e159dfc3b46c8438fc23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1248], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42c6dd67644392e276489449eee12563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f8d25c85a44e159dfc3b46c8438fc23
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_cd85c064f84139709e5828bc3f890507(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd13fadc4eebd76c625f6c5a560c1dd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd85c064f84139709e5828bc3f890507
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_7f2f7462effdeed88a9883bf1ed822ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8df526c8fc1063149a702d33f025e1a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f2f7462effdeed88a9883bf1ed822ea
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_af6b3b8111afc156f7173bf9a2c66d43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0, 1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4ade7430709482162d382ca9722f793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af6b3b8111afc156f7173bf9a2c66d43
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d4ade7430709482162d382ca9722f793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af6b3b8111afc156f7173bf9a2c66d43
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a52803da01243553aab7162decfc8ee7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 18, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1dfb0c519810ff45876df0711725466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a52803da01243553aab7162decfc8ee7
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d1dfb0c519810ff45876df0711725466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a52803da01243553aab7162decfc8ee7
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_88b0e84753c6c210b50f9083ea92e15b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2250f596e0a37d33c9de048968ff3c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88b0e84753c6c210b50f9083ea92e15b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_bfb309cd6aefee1d9e74ce097cc02dd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a7b6d85eb3be8dac2eecb0d310d0252(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfb309cd6aefee1d9e74ce097cc02dd4
    def get_inputs(self):
        return [
            paddle.uniform([2382], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_517c74304efd0b594774ec5de6c83f04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725], dtype='bool'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f78ad941751ec050b5507b376fee3e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_517c74304efd0b594774ec5de6c83f04
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_63ddef6f4358293c2decc43043e52343(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1018da0f0c4fab7dac9e1bcbf5816e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63ddef6f4358293c2decc43043e52343
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2382, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b1018da0f0c4fab7dac9e1bcbf5816e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63ddef6f4358293c2decc43043e52343
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2382, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7be60f617ea0f346d0855a00502b5c79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0206befb92bc6ffc0d21ed8fd328ae5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7be60f617ea0f346d0855a00502b5c79
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0206befb92bc6ffc0d21ed8fd328ae5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7be60f617ea0f346d0855a00502b5c79
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5bcc07e3606cc6c647e94fccd66b55bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd71c97ecff884169a0cfa86daae2090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bcc07e3606cc6c647e94fccd66b55bb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_522f1521ef23687526e764b6fd423dc1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_966f9d5dcc3c698b3ac9238f95ec2843(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_522f1521ef23687526e764b6fd423dc1
    def get_inputs(self):
        return [
            paddle.uniform([2976], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_32e00ed08ee5b71bb3de1b392e36b1d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069], dtype='bool'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_498979ec7346741549d773eaae8deb7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32e00ed08ee5b71bb3de1b392e36b1d1
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_908ac94364b82bae0e9644ff8453e833(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e03805d0cbb8112de1abf51a8ed824d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_908ac94364b82bae0e9644ff8453e833
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2976, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0e03805d0cbb8112de1abf51a8ed824d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_908ac94364b82bae0e9644ff8453e833
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2976, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_af0c7df0ea5f67b80573522f001bbe26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4046a1d803cb52328b730d1e9a6a1bfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af0c7df0ea5f67b80573522f001bbe26
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0e52b54bfd1faf64c02d1acfab7d5259(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5acbf1aaa4d5b94c5b5083d3093445bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e52b54bfd1faf64c02d1acfab7d5259
    def get_inputs(self):
        return [
            paddle.uniform([3753], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_99dc525fb4a5a2ee35e04771a32c173f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581], dtype='bool'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a615824b8fd229dfeed2cd45b9de00ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99dc525fb4a5a2ee35e04771a32c173f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_69701c9001d64c0239c5a8e6725af7f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_544328b210992412e097627f10cc7e4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69701c9001d64c0239c5a8e6725af7f3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3753, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_544328b210992412e097627f10cc7e4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69701c9001d64c0239c5a8e6725af7f3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3753, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_6eff5c556ae4da4fce51bf7187087bc4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb12fc1dc1591840445d2ec8f9d041b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6eff5c556ae4da4fce51bf7187087bc4
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fb12fc1dc1591840445d2ec8f9d041b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6eff5c556ae4da4fce51bf7187087bc4
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b4d2ede7bc56c8b756ee2770197f7610(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f965be4112941509d9c6b41ba58ed90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4d2ede7bc56c8b756ee2770197f7610
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_35aa9f0eaf7afdebc7bfb53a8a6c1b35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27ea582723e99223ba02d30dd83b322a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35aa9f0eaf7afdebc7bfb53a8a6c1b35
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_27ea582723e99223ba02d30dd83b322a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35aa9f0eaf7afdebc7bfb53a8a6c1b35
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1187fb503b25aa311ecbbd3b22cae9c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60545acb49c533412285252116d674c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_02be19272751824395bc441ebe469662(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_963d3acc4bb614b6ce3c1c6f5e9e26c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02be19272751824395bc441ebe469662
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_be41392e596f73db2c387398c8435401(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_657f0e27d53d85066500c1a5d8ac158d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be41392e596f73db2c387398c8435401
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_3402ff1c3e7c4163c26bf33200bb668b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 480], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed0d89523c2474b740bac57dac5cfbf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3402ff1c3e7c4163c26bf33200bb668b
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_4a7fadbacbeb6a0169e8c357384e2604(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63a0a7df99ff9d70740a853da17154b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a7fadbacbeb6a0169e8c357384e2604
    def get_inputs(self):
        return [
            paddle.uniform([2, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ea167d8a3d25adc8e469cb8fe431b061(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e9113e06b996ee0b0d398a531995d5f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea167d8a3d25adc8e469cb8fe431b061
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26883453130722046, 0.027636855840682983], dtype='float32').reshape([2]),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_64e6e2abe42a322dae8aa4d5d80c50be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8d50ef1c8d61103dfd08ba66fbc37a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e6e2abe42a322dae8aa4d5d80c50be
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_8fd71657c5a3295d87f9dce9a4cf9227(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_017b6c17faf4571f776b930476e0c2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fd71657c5a3295d87f9dce9a4cf9227
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_eaa28b9c9fdbd4de6a76c6c610984b97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2adad0f21bb00bb0c3daa9a3427c82bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eaa28b9c9fdbd4de6a76c6c610984b97
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6c86fdbd166a3260d4db00be9b5c2a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_468324d038ef45fe9a7bd877c01e86c4
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e32ae56b0c730d0663d615afcd4265f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f05204689c07bb789ca594e9397fced2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e32ae56b0c730d0663d615afcd4265f1
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a992dabc80f67e36e1bea277006745f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e32ae56b0c730d0663d615afcd4265f1
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7268e225bd868c064ddf9216d0b86791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a61ef7c1fa5d4f90d4658a7ac403e6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_963d3acc4bb614b6ce3c1c6f5e9e26c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02be19272751824395bc441ebe469662
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_16105083f5955bb4f70c5b160678f050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c16b8ff6fe14839d82f66dd890ab46a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_97f6f280fd70329e40fdae17f87a9dc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0962c24d2d088cd2536017e2a335c691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97f6f280fd70329e40fdae17f87a9dc3
    def get_inputs(self):
        return [
            paddle.uniform([1995], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_abc31e9cac760e11f68619b77f8f59d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_572bd22ae5f076d02d61bd33e4fcd0f6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_17f7d9ddbd0aa80aed7e7e351c18d9a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5741486f794fe8df4d634da2985fd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17f7d9ddbd0aa80aed7e7e351c18d9a1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1995, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a5741486f794fe8df4d634da2985fd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17f7d9ddbd0aa80aed7e7e351c18d9a1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1995, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7adf9e1645f0b1751954ea5b989121d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ae0b024347c42d115b1e0762b868717
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_84a315f151cc05d6c697931fee6a0766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57094ea9c93330fe82f1162ee7f22407
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0ae8e870fdd50e7d2c89e99461e0648e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba9a8f4314cc1e56616a5ba163e6f5eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ae8e870fdd50e7d2c89e99461e0648e
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_22a1e3568ec7e399fff742394f615dc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_024c4f0c07e3ee74425f812dbbddc555
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_3481ee2c2871bc70dcde4bdfc90bd1ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a0d575e151456143457986d73cf4d7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3481ee2c2871bc70dcde4bdfc90bd1ef
    def get_inputs(self):
        return [
            paddle.uniform([150, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a0b34a63deccd6fee29ffede439374a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fb239ccc64524c6f82818c2991c18e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0b34a63deccd6fee29ffede439374a5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4933174e1265ffa789d862f944fb5cb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_190ceafb46a05bfe930fe126a7bb29f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4933174e1265ffa789d862f944fb5cb6
    def get_inputs(self):
        return [
            paddle.uniform([4185], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_738bc92519df1769249fd2c751c7a354(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400], dtype='bool'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_05f64c386153123f2c2fdf992066696f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_738bc92519df1769249fd2c751c7a354
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1d359eac72b4ba8c8077e1fc1b2d68a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f2a80549d8531dcdb2ea7bf47e3a511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d359eac72b4ba8c8077e1fc1b2d68a6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4185, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8f2a80549d8531dcdb2ea7bf47e3a511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d359eac72b4ba8c8077e1fc1b2d68a6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4185, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_60d993bf59e60e3d4f15a53b0a9ef924(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 624], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbb41e55496920d4d607eb431983c7ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60d993bf59e60e3d4f15a53b0a9ef924
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_017b6c17faf4571f776b930476e0c2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fd71657c5a3295d87f9dce9a4cf9227
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()