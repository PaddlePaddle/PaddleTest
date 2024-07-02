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


class PrimitiveOp_96d05d289c3e0b010880ca32558508ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33754b314933c5348f0d164df72c1735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96d05d289c3e0b010880ca32558508ad
    def get_inputs(self):
        return [
            paddle.uniform([1787, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c52d9ce53b7b10bf96c50c2e5c259af0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d581d4a7b1a4ddc946563bdcf909282
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.01077839732170105, 0.23139362037181854, -0.36757537722587585, 0.2106194794178009], [-0.21976801753044128, 0.23060455918312073, 0.12467886507511139, 0.017615854740142822], [-0.2861097455024719, -0.2822428345680237, -0.26744967699050903, 0.4451574385166168], [0.054540008306503296, 0.15870381891727448, -0.018163494765758514, -0.011783301830291748], [-0.16896668076515198, -0.45631930232048035, 0.1181577667593956, 0.26534923911094666]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_fe49d0ad0c9384e001e6a26784cbdb0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d581d4a7b1a4ddc946563bdcf909282
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03768511861562729, 0.3141765892505646, 0.09385469555854797, 0.18678498268127441], [0.08251011371612549, 0.06584697961807251, -0.11500242352485657, -0.17597244679927826], [-0.21548722684383392, 0.15471479296684265, 0.2636128067970276, -0.06203708052635193], [0.08251011371612549, 0.06584697961807251, -0.11500242352485657, -0.17597244679927826], [-0.21548722684383392, 0.15471479296684265, 0.2636128067970276, -0.06203708052635193]], dtype='float32').reshape([5, 4]),
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


class PrimitiveOp_da71509bd0786c3ff639ca0776adcb60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d0ccd5d7a546f88e0497f1aa6b32b7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da71509bd0786c3ff639ca0776adcb60
    def get_inputs(self):
        return [
            paddle.uniform([5585, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_23f862928a4a6b04a3be4e49c0af1b6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2178e9c120dbd80e3cc79426b0ee0010
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.05574280023574829, -0.19090011715888977, 0.07926230132579803, 0.08707726001739502], [-0.07147979736328125, -0.2017761468887329, -0.04350680112838745, -0.36970213055610657], [-0.30316054821014404, 0.08338502049446106, -0.24412307143211365, 0.23371408879756927], [-0.07147979736328125, -0.2017761468887329, -0.04350680112838745, -0.36970213055610657], [-0.30316054821014404, 0.08338502049446106, -0.24412307143211365, 0.23371408879756927], [-0.09875491261482239, -0.18579769134521484, -0.22553396224975586, -0.21217313408851624], [-0.09875491261482239, -0.18579769134521484, -0.22553396224975586, -0.21217313408851624]], dtype='float32').reshape([7, 4]),
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


class PrimitiveOp_322409d58ace4b4ad3ac84e1a472499d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ac98c3ba545edca4bf1d2f4ec2f9305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_322409d58ace4b4ad3ac84e1a472499d
    def get_inputs(self):
        return [
            paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4a3c7d43bb4eadaeeb041e0cb2ec1b12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1cbae27237eea6835a53b49e94203312(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a3c7d43bb4eadaeeb041e0cb2ec1b12
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_16561a9f8859355a72d762300e2a9773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b62f9555ad9c7c4f66aa0f0f18d80500
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19524270296096802, 0.11190202832221985, 0.24526070058345795, 0.04938575625419617], [0.2459895759820938, -0.23118141293525696, -0.38403549790382385, -0.043900057673454285], [0.04088515788316727, 0.16825607419013977, -0.2890157401561737, 0.1842956840991974], [-0.08379620313644409, -0.3274226784706116, -0.012366145849227905, -0.11941033601760864], [-0.08379620313644409, -0.3274226784706116, -0.012366145849227905, -0.11941033601760864], [0.04088515788316727, 0.16825607419013977, -0.2890157401561737, 0.1842956840991974]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_3b83819ea5772ed4d067b66742c08c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d581d4a7b1a4ddc946563bdcf909282
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.011681139469146729, -0.4303775131702423, -0.43140000104904175, 0.0924951583147049], [-0.025084972381591797, 0.10988263785839081, -0.12713006138801575, -0.03839513659477234], [-0.10030119121074677, 0.0020422935485839844, -0.024833127856254578, 0.0032770633697509766], [0.08440111577510834, 0.24669235944747925, 0.39913398027420044, 0.20100060105323792], [-0.011681139469146729, -0.4303775131702423, -0.43140000104904175, 0.0924951583147049]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_36278db0405469fba5b71278c72830cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_614f5b6dd691404e7c87db643cbe5a80
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.14627555012702942, -0.13115960359573364, -0.14302843809127808, -0.10098734498023987], [0.23363037407398224, -0.13121597468852997, -0.03195944428443909, 0.3712884783744812], [-0.20839068293571472, 0.00977557897567749, 0.09457176923751831, -0.17659586668014526], [0.14074213802814484, -0.15372057259082794, 0.018863890320062637, 0.19768190383911133]], dtype='float32').reshape([4, 4]),
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


class PrimitiveOp_e5e8cd825508d942836d558d8d82dc7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_742a3639e0c92b43f111922bd1413d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5e8cd825508d942836d558d8d82dc7a
    def get_inputs(self):
        return [
            paddle.uniform([2049, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_390c3e40276c83825805aae0ea3ad0b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2178e9c120dbd80e3cc79426b0ee0010
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.27607500553131104, 0.16805040836334229, 0.01612101122736931, 0.20125778019428253], [-0.27607500553131104, 0.16805040836334229, 0.01612101122736931, 0.20125778019428253], [0.2777993977069855, 0.1656722128391266, -0.0767522007226944, 0.1466934084892273], [0.40827086567878723, 0.16954927146434784, 0.006458401679992676, 0.31475168466567993], [-0.25782519578933716, -0.012537658214569092, -0.1426226645708084, 0.1456732600927353], [0.03390437364578247, 0.029349535703659058, -0.15627236664295197, 0.2299584001302719], [-0.03064623475074768, -0.18180377781391144, 0.06553564965724945, -0.022069208323955536]], dtype='float32').reshape([7, 4]),
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


class PrimitiveOp_6d98c0080809d9e09048ff8fd6267b2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07bece73ff44d0bb9e556a246b14c474(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d98c0080809d9e09048ff8fd6267b2a
    def get_inputs(self):
        return [
            paddle.uniform([4634, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b71565a075b38fe4f58e086bab6abf2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c97ce101b32b4d96a81cc9ed7fdf49a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b71565a075b38fe4f58e086bab6abf2f
    def get_inputs(self):
        return [
            paddle.uniform([1000, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33940cc1bbad5b384fc14a6430e37cfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b62f9555ad9c7c4f66aa0f0f18d80500
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.419389009475708, -0.2385486662387848, 0.17910200357437134, 0.25541141629219055], [0.15476015210151672, 0.12787313759326935, -0.274046927690506, 0.007582925260066986], [0.15476015210151672, 0.12787313759326935, -0.274046927690506, 0.007582925260066986], [-0.19365759193897247, 0.06694699078798294, -0.38479316234588623, 0.05158597230911255], [0.36040982604026794, 0.0010609328746795654, 0.10762536525726318, -0.2891213893890381], [-0.16834193468093872, -0.3301253020763397, -0.2282804548740387, -0.03077937290072441]], dtype='float32').reshape([6, 4]),
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


class PrimitiveOp_1be9b1a10f597a1a4d5f8315a0971274(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_978379a3ec596331f08b73e7cca5a68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1be9b1a10f597a1a4d5f8315a0971274
    def get_inputs(self):
        return [
            paddle.uniform([2382, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6647202d00a93fff6e8564c568564f4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8aedf69785a81c0254d4e5e73924606d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6647202d00a93fff6e8564c568564f4d
    def get_inputs(self):
        return [
            paddle.uniform([2976, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f4d06ad2ac0c9951ca876da1d0d29410(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ccb2e4a44f4b95983ac5fffe411ea058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4d06ad2ac0c9951ca876da1d0d29410
    def get_inputs(self):
        return [
            paddle.uniform([3753, 4], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_f80cd217b271f35e9b09f62e243db1ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_195680e9733ad9ef130d71a5df6fc53c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f80cd217b271f35e9b09f62e243db1ca
    def get_inputs(self):
        return [
            paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f409ce11b0c845a86b281488cfb73f3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d581d4a7b1a4ddc946563bdcf909282
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05009490251541138, 0.15385662019252777, -0.2759556174278259, 0.3261295258998871], [-0.1172669380903244, -0.12045417726039886, 0.021406829357147217, -0.13164407014846802], [-0.02002665400505066, 0.08511902391910553, 0.2713954448699951, -0.060312606394290924], [-0.02002665400505066, 0.08511902391910553, 0.2713954448699951, -0.060312606394290924], [0.3452134132385254, 0.04304805397987366, -0.06859937310218811, 0.2436056286096573]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_53e1bb213b06054c86cb932c51563ae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2178e9c120dbd80e3cc79426b0ee0010
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10310405492782593, 0.2407267689704895, 0.1336056888103485, -0.3315361738204956], [-0.4141213595867157, 0.22854921221733093, -0.3232783377170563, -0.018062174320220947], [-0.1820223480463028, 0.009562350809574127, 0.15676060318946838, -0.008004188537597656], [0.10310405492782593, 0.2407267689704895, 0.1336056888103485, -0.3315361738204956], [0.37853386998176575, 0.035849153995513916, -0.40783774852752686, 0.12761946022510529], [0.1093188226222992, 0.21362468600273132, 0.153514102101326, -0.08548074960708618], [0.37853386998176575, 0.035849153995513916, -0.40783774852752686, 0.12761946022510529]], dtype='float32').reshape([7, 4]),
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